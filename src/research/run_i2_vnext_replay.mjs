#!/usr/bin/env node
import fs from 'node:fs';
import { simulateFullSecondInning } from '../model/i2_inning_model.js';
import { predictI2EventVector } from '../model/i2_vnext_event_model.js';
import { applyEnvironmentalEventVector } from '../event_probability_engine.js';
import { createSeededRandom, seedFromGameId } from '../model/seeded_random.js';

function arg(name, fallback=null) {
  const i=process.argv.indexOf(name);
  return i>=0 ? process.argv[i+1] : fallback;
}
const INPUT=arg('--input','data/derived/i2_vnext/replay_2025_inputs.json');
const MODEL=arg('--model','data/derived/i2_vnext/i2_vnext_event_model.json');
const ARSENAL=arg('--arsenal','data/derived/i2_vnext/arsenal_profile_2024.json');
const PARKS=arg('--parks','data/derived/i2_vnext/park/savant_venue_profiles_2024_3yr.json');
const PLAY=arg('--play-calibration','data/derived/i2/i2_play_calibration.json');
const OUTPUT=arg('--output','data/derived/i2_vnext/replay_2025_predictions.json');
const TRIALS=Number(arg('--trials','10000'));

if (!Number.isInteger(TRIALS) || TRIALS < 1000) throw new Error('--trials must be an integer >= 1000');

const replay=JSON.parse(fs.readFileSync(INPUT,'utf8'));
const model=JSON.parse(fs.readFileSync(MODEL,'utf8'));
const arsenal=JSON.parse(fs.readFileSync(ARSENAL,'utf8'));
const playCalibration=JSON.parse(fs.readFileSync(PLAY,'utf8'));
const parkProfiles=fs.existsSync(PARKS) ? JSON.parse(fs.readFileSync(PARKS,'utf8')) : [];

const RETRO_TEAM_ALIASES = {
  ARI:['ARI','ARIZONA DIAMONDBACKS'], ATL:['ATL','ATLANTA BRAVES'],
  BAL:['BAL','BALTIMORE ORIOLES'], BOS:['BOS','BOSTON RED SOX'],
  CHA:['CHW','CWS','CHICAGO WHITE SOX'], CHN:['CHC','CHICAGO CUBS'],
  CIN:['CIN','CINCINNATI REDS'], CLE:['CLE','CLEVELAND GUARDIANS'],
  COL:['COL','COLORADO ROCKIES'], DET:['DET','DETROIT TIGERS'],
  HOU:['HOU','HOUSTON ASTROS'], KCA:['KC','KCR','KANSAS CITY ROYALS'],
  LAA:['LAA','LOS ANGELES ANGELS'], LAN:['LAD','LOS ANGELES DODGERS'],
  MIA:['MIA','MIAMI MARLINS'], MIL:['MIL','MILWAUKEE BREWERS'],
  MIN:['MIN','MINNESOTA TWINS'], NYA:['NYY','NEW YORK YANKEES'],
  NYN:['NYM','NEW YORK METS'], PHI:['PHI','PHILADELPHIA PHILLIES'],
  PIT:['PIT','PITTSBURGH PIRATES'], SDN:['SD','SDP','SAN DIEGO PADRES'],
  SEA:['SEA','SEATTLE MARINERS'], SFN:['SF','SFG','SAN FRANCISCO GIANTS'],
  SLN:['STL','ST. LOUIS CARDINALS','ST LOUIS CARDINALS'],
  TEX:['TEX','TEXAS RANGERS'], TOR:['TOR','TORONTO BLUE JAYS'],
  WAS:['WSH','WAS','WASHINGTON NATIONALS'],
};

function norm(x){ return String(x??'').trim().toUpperCase(); }

function venueFor(game){
  // 2025 Athletics (Sutter Health Park) and Rays (Steinbrenner Field) did not
  // have an established 2024 Savant park profile. Deliberately use neutral.
  if (['ATH','OAK','TBA'].includes(norm(game.home_team_retro))) {
    return {profile:null,status:'NEW_OR_TEMPORARY_2025_VENUE_NEUTRAL'};
  }
  const aliases=RETRO_TEAM_ALIASES[norm(game.home_team_retro)] || [];
  const profile=parkProfiles.find(p=>aliases.includes(norm(p.team)));
  return profile
    ? {profile,status:'PRIOR_SEASON_SAVANT_3YR'}
    : {profile:null,status:'UNMATCHED_PRIOR_SEASON_VENUE_NEUTRAL'};
}

function stand(bats, throws){
  const b=norm(bats), t=norm(throws);
  if (b==='B') return t==='L' ? 'R' : 'L';
  return b==='L' ? 'L' : 'R';
}

function makeLineup(rows, leagueRates){
  return rows.map(x=>({
    id:Number(x.mlbam),
    bats:x.bats,
    side:x.bats,
    eventRates:leagueRates,
  }));
}
function makePitcher(x, leagueRates){
  return {
    id:Number(x.mlbam),
    throws:x.throws,
    eventRatesAllowed:leagueRates,
  };
}
function loglossTerm(y,p){
  const q=Math.max(1e-9,Math.min(1-1e-9,p));
  return -(y*Math.log(q)+(1-y)*Math.log(1-q));
}

const leagueRates=replay?.i1_state_model?.event_rates;
if (!leagueRates) throw new Error('Replay input missing i1_state_model.event_rates');

const predictions=[];
let brier=0, ll=0, parkMatched=0;
for (const game of replay.games) {
  const venue=venueFor(game);
  if (venue.profile) parkMatched += 1;
  const away={
    lineup:makeLineup(game.away_lineup,leagueRates),
    starter:makePitcher(game.away_starter,leagueRates),
  };
  const home={
    lineup:makeLineup(game.home_lineup,leagueRates),
    starter:makePitcher(game.home_starter,leagueRates),
  };
  const provider=({batter,pitcher})=>{
    const batterSide=stand(batter.bats,pitcher.throws);
    const neutral=predictI2EventVector({
      batterId:batter.id,
      pitcherId:pitcher.id,
      batterSide,
      pitcherThrows:pitcher.throws,
      model,
      arsenalProfile:arsenal,
    });
    return applyEnvironmentalEventVector({
      neutralVector:neutral,
      environmentalContext:venue.profile,
      batterSide,
    }).probabilities;
  };

  const random=createSeededRandom(seedFromGameId(game.gid,2025));
  const result=simulateFullSecondInning({
    away,home,league:leagueRates,environmentalContext:null,
    weights:{batter:0.5,pitcher:0.5},
    trials:TRIALS,random,playCalibration,
    i2EventVectorProvider:provider,
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
    starter_continuation_audit:game.audit,
  });
}

const n=predictions.length;
const payload={
  version:'i2-vnext-full-replay-v1',
  generated_at:new Date().toISOString(),
  season:2025,
  model_version:model.version,
  model_training:model.training,
  holdout_policy:model.holdout_policy,
  trials_per_game:TRIALS,
  market_inputs_used:false,
  observed_i2_state_used_as_predictor:false,
  i1_state_model:replay.i1_state_model,
  park_rule:'prior-season Savant 3yr profile; explicit neutral for new/temporary or unmatched venue',
  park_match_rate:n ? parkMatched/n : null,
  n,
  raw_brier:n ? brier/n : null,
  raw_logloss:n ? ll/n : null,
  predictions,
};
fs.mkdirSync(new URL('../../data/derived/i2_vnext/',import.meta.url),{recursive:true});
fs.writeFileSync(OUTPUT,JSON.stringify(payload));
console.log(JSON.stringify({
  n:payload.n,raw_brier:payload.raw_brier,raw_logloss:payload.raw_logloss,
  park_match_rate:payload.park_match_rate,trials_per_game:TRIALS,
},null,2));
