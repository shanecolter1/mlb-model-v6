#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { simulateFullSecondInning } from '../model/i2_inning_model.js';
import { predictI2EventVector } from '../model/i2_vnext_event_model.js';
import { applyEnvironmentalEventVector, buildNeutralEventVector } from '../event_probability_engine.js';
import { createSeededRandom, seedFromGameId } from '../model/seeded_random.js';
import { venueForReplayGame } from './i2_vnext_replay_venue.mjs';
import { predictI2EventVectorResearch } from './i2_vnext_feature_ablation_provider.mjs';

function arg(name, fallback=null) {
  const i=process.argv.indexOf(name);
  return i>=0 ? process.argv[i+1] : fallback;
}
const INPUT=arg('--input','data/derived/i2_vnext/replay_2025_inputs.json');
const MODEL=arg('--model','data/derived/i2_vnext/i2_vnext_event_model.json');
const WALKFORWARD=arg('--walkforward-manifest','data/derived/i2_vnext/walkforward_2025/manifest.json');
const ARSENAL=arg('--arsenal','data/derived/i2_vnext/arsenal_profile_2024.json');
const PARKS=arg('--parks','data/derived/i2_vnext/park/savant_venue_profiles_2024_3yr.json');
const PLAY=arg('--play-calibration','data/derived/model_calibration/seasonal/production_pa_transition_table_shrunk.json');
const OUTPUT=arg('--output','data/derived/i2_vnext/replay_2025_predictions.json');
const TRIALS=Number(arg('--trials','10000'));
const I2_MODEL=arg('--i2-model','vnext');
if (!['vnext','production_formula'].includes(I2_MODEL)) throw new Error('Unknown --i2-model');
const I2_FEATURE_MODE=arg('--i2-feature-mode','baseline');
if (!['baseline','no_platoon'].includes(I2_FEATURE_MODE)) throw new Error('Unknown --i2-feature-mode');
const I1_MODE=arg('--i1-mode','league');
const I1_PLAYER_RATES=arg('--i1-player-rates','active');
if (!['active','neutralized'].includes(I1_PLAYER_RATES)) throw new Error('--i1-player-rates must be active or neutralized');
const I1_ENVIRONMENT=arg('--i1-environment','neutral');
if (!['neutral','prior_season_park'].includes(I1_ENVIRONMENT)) throw new Error('Unknown --i1-environment');
const SHARD_COUNT=Number(arg('--shard-count','1'));
const SHARD_INDEX=Number(arg('--shard-index','0'));
if (!['league','player_asof'].includes(I1_MODE)) throw new Error('--i1-mode must be league or player_asof');

if (!Number.isInteger(TRIALS) || TRIALS < 1000) throw new Error('--trials must be an integer >= 1000');
if (!Number.isInteger(SHARD_COUNT) || SHARD_COUNT < 1) throw new Error('--shard-count must be an integer >= 1');
if (!Number.isInteger(SHARD_INDEX) || SHARD_INDEX < 0 || SHARD_INDEX >= SHARD_COUNT) {
  throw new Error('--shard-index must be an integer in [0, shard-count)');
}

const replay=JSON.parse(fs.readFileSync(INPUT,'utf8'));
const REPLAY_SEASON=Number(replay?.season);
const model=JSON.parse(fs.readFileSync(MODEL,'utf8'));
const walkforward=fs.existsSync(WALKFORWARD)
  ? JSON.parse(fs.readFileSync(WALKFORWARD,'utf8'))
  : null;
const walkforwardModels=new Map();
if (walkforward) {
  if (walkforward.market_inputs_used !== false) throw new Error('Walk-forward manifest is not market-isolated');
  const root=path.dirname(WALKFORWARD);
  for (const entry of walkforward.entries || []) {
    const modelPath=path.join(root,entry.model_file);
    const artifact=JSON.parse(fs.readFileSync(modelPath,'utf8'));
    if (artifact.market_inputs_used !== false) throw new Error(`Walk-forward model not market-isolated: ${entry.model_file}`);
    walkforwardModels.set(entry.model_file,artifact);
  }
}
const arsenal=JSON.parse(fs.readFileSync(ARSENAL,'utf8'));
const rawPlayCalibration=JSON.parse(fs.readFileSync(PLAY,'utf8'));
function adaptPlayCalibration(payload){
  if (payload?.base_transitions) return payload;
  if (!payload?.states) throw new Error('Unsupported transition artifact');
  const eventMap={
    strikeout:'out',
    ball_in_play_out:'out',
    walk:'bb',
    hit_by_pitch:'bb',
    single:'single',
    double:'double',
    triple:'triple',
    home_run:'hr',
  };
  const base_transitions={};
  for (const [event,source] of Object.entries(eventMap)) {
    for (let outs=0; outs<3; outs+=1) {
      for (let mask=0; mask<8; mask+=1) {
        base_transitions[`${event}|${outs}|${mask}`]=payload.states[`${event}|${outs}|${mask}`] || payload.states[`${source}|${outs}|${mask}`] || [];
      }
    }
  }
  return {
    version:`adapted-${payload.version || 'validated-transitions'}`,
    base_transitions,
    pitch_count_pmf:{},
    governance:{
      sourceTrainingYears:payload.training_years || null,
      selectionYear:payload.selection_year || null,
      lockedValidationYear:payload.locked_validation_year || null,
    },
  };
}
const playCalibration=adaptPlayCalibration(rawPlayCalibration);
const parkProfiles=fs.existsSync(PARKS) ? JSON.parse(fs.readFileSync(PARKS,'utf8')) : [];
if (I1_ENVIRONMENT==='prior_season_park' && (!Array.isArray(parkProfiles) || !parkProfiles.length)) {
  throw new Error('Park-aware I1 replay requires prior-season park profiles');
}

function norm(x){ return String(x??'').trim().toUpperCase(); }
function compactDate(x){ return String(x??'').replace(/[^0-9]/g,'').slice(0,8); }
function modelForGame(game){
  if (!walkforward) return {artifact:model,entry:null};
  const d=compactDate(game.date);
  let chosen=null;
  for (const entry of walkforward.entries || []) {
    if (compactDate(entry.effective_from) <= d) {
      if (!chosen || compactDate(entry.effective_from) > compactDate(chosen.effective_from)) chosen=entry;
    }
  }
  if (!chosen) throw new Error(`No walk-forward model available for ${game.gid} on ${game.date}`);
  const artifact=walkforwardModels.get(chosen.model_file);
  if (!artifact) throw new Error(`Missing walk-forward model artifact: ${chosen.model_file}`);
  const maxTraining=compactDate(artifact?.training?.end);
  if (maxTraining && maxTraining >= d) {
    throw new Error(`Point-in-time leakage: model training end ${artifact.training.end} >= game ${game.date}`);
  }
  return {artifact,entry:chosen};
}

function stand(bats, throws){
  const b=norm(bats), t=norm(throws);
  if (b==='B') return t==='L' ? 'R' : 'L';
  return b==='L' ? 'L' : 'R';
}

function requireAsOfRates(x, label){
  const rates=x?.i1_event_rates_asof;
  if (!rates || typeof rates !== 'object') {
    throw new Error(`Missing leakage-safe player-asof I1 rates for ${label}`);
  }
  for (const key of ['single','double','triple','home_run','walk','hit_by_pitch','strikeout','ball_in_play_out']) {
    const value=Number(rates[key]);
    if (!Number.isFinite(value) || value < 0) {
      throw new Error(`Invalid player-asof I1 rate ${key} for ${label}`);
    }
  }
  return rates;
}
function makeLineup(rows, leagueRates){
  return rows.map((x,i)=>({
    id:Number(x.mlbam),
    bats:x.bats,
    side:norm(x.bats)==='B' ? 'S' : x.bats,
    eventRates:I1_MODE === 'player_asof'
      ? requireAsOfRates(x,`hitter ${x.mlbam || i}`)
      : leagueRates,
  }));
}
function makePitcher(x, leagueRates){
  return {
    id:Number(x.mlbam),
    throws:x.throws,
    eventRatesAllowed:I1_MODE === 'player_asof'
      ? requireAsOfRates(x,`pitcher ${x.mlbam}`)
      : leagueRates,
  };
}
function loglossTerm(y,p){
  const q=Math.max(1e-9,Math.min(1-1e-9,p));
  return -(y*Math.log(q)+(1-y)*Math.log(1-q));
}

const leagueRates=replay?.i1_state_model?.event_rates;
if (!leagueRates) throw new Error('Replay input missing i1_state_model.event_rates');
if (I1_MODE === 'player_asof') {
  if (replay?.i1_state_model?.player_specific_i1_talent_used !== true) {
    throw new Error('Replay input is not governed for player-specific I1 talent');
  }
  if (!replay?.i1_player_asof_model) {
    throw new Error('Replay input missing i1_player_asof_model governance');
  }
}

const replayGames=(replay.games || []).filter((_,index)=>index % SHARD_COUNT === SHARD_INDEX);
const predictions=[];
let brier=0, ll=0, parkMatched=0;
for (const game of replayGames) {
  const {artifact:activeModel,entry:modelEntry}=modelForGame(game);
  const venue=venueForReplayGame(game,parkProfiles,REPLAY_SEASON);
  if (venue.profile) parkMatched += 1;
  const away={
    lineup:makeLineup(game.away_lineup,leagueRates),
    starter:makePitcher(game.away_starter,leagueRates),
  };
  const home={
    lineup:makeLineup(game.home_lineup,leagueRates),
    starter:makePitcher(game.home_starter,leagueRates),
  };
  const formulaCache=new Map();
  const provider=({batter,pitcher})=>{
    const batterSide=stand(batter.bats,pitcher.throws);
    if (I2_MODEL === 'production_formula') {
      const key=`${batter.id}|${pitcher.id}|${batterSide}`;
      if (!formulaCache.has(key)) {
        const neutralVector=buildNeutralEventVector({batter:batter.eventRates,pitcher:pitcher.eventRatesAllowed,league:leagueRates,weights:{batter:0.5,pitcher:0.5}});
        formulaCache.set(key,applyEnvironmentalEventVector({neutralVector,environmentalContext:venue.profile,batterSide}).probabilities);
      }
      return formulaCache.get(key);
    }
    const neutral=I2_FEATURE_MODE === 'baseline'
      ? predictI2EventVector({
          batterId:batter.id,
          pitcherId:pitcher.id,
          batterSide,
          pitcherThrows:pitcher.throws,
          model:activeModel,
          arsenalProfile:arsenal,
        })
      : predictI2EventVectorResearch({
          batterId:batter.id,
          pitcherId:pitcher.id,
          batterSide,
          pitcherThrows:pitcher.throws,
          model:activeModel,
          arsenalProfile:arsenal,
          mode:I2_FEATURE_MODE,
        });
    return applyEnvironmentalEventVector({
      neutralVector:neutral,
      environmentalContext:venue.profile,
      batterSide,
    }).probabilities;
  };

  const random=createSeededRandom(seedFromGameId(game.gid,REPLAY_SEASON));
  const result=simulateFullSecondInning({
    away,home,league:leagueRates,
    environmentalContext:I1_ENVIRONMENT==='prior_season_park' ? venue.profile : null,
    weights:{batter:0.5,pitcher:0.5},
    trials:TRIALS,random,playCalibration,
    i2EventVectorProvider:provider,
    neutralizeI1PlayerRates:I1_PLAYER_RATES === 'neutralized',
  });
  const y=Number(game.observed.under05);
  const p=Number(result.under05);
  brier += (p-y)**2;
  ll += loglossTerm(y,p);
  predictions.push({
    gid:game.gid,date:game.date,
    away_team_retro:game.away_team_retro,
    home_team_retro:game.home_team_retro,
    observed_under05:y,
    observed_full_i2_runs:game.observed.full_i2_runs,
    raw_under05:p,
    raw_over05:1-p,
    top2_score_probability:result.top2.cumulative['1+'],
    bottom2_score_probability:result.bottom2.cumulative['1+'],
    park_status:venue.status,
    model_effective_from:modelEntry?.effective_from || null,
    model_training_end:activeModel?.training?.end || null,
    starter_continuation_audit:game.audit,
  });
}

const n=predictions.length;
const payload={
  version:'i2-vnext-full-replay-v3-player-asof-i1',
  generated_at:new Date().toISOString(),
  season:REPLAY_SEASON,
  model_version:walkforward?.version || model.version,
  base_model_version:model.version,
  model_training:model.training,
  holdout_policy:model.holdout_policy,
  walkforward_policy:walkforward ? {
    point_in_time:true,
    cadence:walkforward.cadence,
    policy:walkforward.policy,
    hyperparameters:walkforward.hyperparameters,
    entries:walkforward.entries,
  } : {
    point_in_time:false,
    cadence:null,
    policy:'single static model',
  },
  trials_per_game:TRIALS,
  replay_games_total:Array.isArray(replay.games) ? replay.games.length : 0,
  shard_count:SHARD_COUNT,
  shard_index:SHARD_INDEX,
  shard_games:replayGames.length,
  market_inputs_used:false,
  observed_i2_state_used_as_predictor:false,
  point_in_time_player_refits:Boolean(walkforward),
  i2_model:I2_MODEL,
  i2_feature_mode:I2_FEATURE_MODE,
  comparison_scope:I2_MODEL === 'production_formula' ? 'Controlled production event formula; shared pregame as-of rates, prior-season baseline, I1 state and transitions. Not exact deployed production.' : null,
  i1_state_mode:I1_MODE,
  i1_player_rates:I1_PLAYER_RATES,
  i1_environment:I1_ENVIRONMENT,
  i1_state_model:replay.i1_state_model,
  i1_player_asof_model:I1_MODE === 'player_asof' ? (replay.i1_player_asof_model || null) : null,
  park_rule:'prior-season Savant 3yr profile; explicit neutral for new/temporary or unmatched venue',
  park_match_rate:n ? parkMatched/n : null,
  transition_model:playCalibration.version,
  transition_governance:playCalibration.governance || null,
  n,
  raw_brier:n ? brier/n : null,
  raw_logloss:n ? ll/n : null,
  predictions,
};
fs.mkdirSync(path.dirname(OUTPUT),{recursive:true});
fs.writeFileSync(OUTPUT,JSON.stringify(payload));
console.log(JSON.stringify({
  n:payload.n,raw_brier:payload.raw_brier,raw_logloss:payload.raw_logloss,
  park_match_rate:payload.park_match_rate,trials_per_game:TRIALS,
  replay_games_total:payload.replay_games_total,shard_count:SHARD_COUNT,shard_index:SHARD_INDEX,
  point_in_time_player_refits:payload.point_in_time_player_refits,
  i2_feature_mode:payload.i2_feature_mode,
  i1_state_mode:payload.i1_state_mode,
  i1_player_rates:payload.i1_player_rates,
  i1_environment:payload.i1_environment,
},null,2));
