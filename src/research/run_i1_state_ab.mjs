#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { simulateHalfInningWithLineup } from '../model/i2_inning_model.js';
import { createSeededRandom, seedFromGameId } from '../model/seeded_random.js';

function arg(name, fallback=null) {
  const i=process.argv.indexOf(name);
  return i>=0 ? process.argv[i+1] : fallback;
}
const INPUT=arg('--input','data/derived/i2_vnext/i1_state_ab_2024_inputs.json');
const PLAY=arg('--play-calibration','data/derived/model_calibration/seasonal/production_pa_transition_table_shrunk.json');
const OUTPUT=arg('--output','data/derived/i2_vnext/i1_state_ab_2024_rows.json');
const TRIALS=Number(arg('--trials','2500'));
if (!Number.isInteger(TRIALS) || TRIALS < 1000) throw new Error('--trials must be >=1000');

const input=JSON.parse(fs.readFileSync(INPUT,'utf8'));
const rawPlay=JSON.parse(fs.readFileSync(PLAY,'utf8'));
function adaptPlayCalibration(payload){
  if (payload?.base_transitions) return payload;
  if (!payload?.states) throw new Error('Unsupported transition artifact');
  const map={strikeout:'out',ball_in_play_out:'out',walk:'bb',hit_by_pitch:'bb',single:'single',double:'double',triple:'triple',home_run:'hr'};
  const base_transitions={};
  for (const [event,source] of Object.entries(map)) {
    for (let outs=0;outs<3;outs++) for (let mask=0;mask<8;mask++) {
      base_transitions[`${event}|${outs}|${mask}`]=payload.states[`${source}|${outs}|${mask}`] || [];
    }
  }
  return {version:`adapted-${payload.version||'validated'}`,base_transitions,pitch_count_pmf:{}};
}
const playCalibration=adaptPlayCalibration(rawPlay);
const league=input.league_event_rates;

function lineup(rows, mode){
  return rows.map(x=>({
    id:x.id,
    side:'R',
    eventRates:mode==='player_asof' ? x.i1_event_rates_asof : league,
  }));
}
function pitcher(x, mode){
  return {
    id:x.id,
    throws:'R',
    eventRatesAllowed:mode==='player_asof' ? x.i1_event_rates_asof : league,
  };
}
function distribution(game, side, mode){
  const isTop=side==='top';
  const lu=lineup(isTop?game.away_lineup:game.home_lineup,mode);
  const p=pitcher(isTop?game.home_starter:game.away_starter,mode);
  const counts=Array(10).fill(0);
  const random=createSeededRandom(seedFromGameId(`${game.gid}:${side}`, mode==='league'?2401:2402));
  for(let i=0;i<TRIALS;i++){
    const r=simulateHalfInningWithLineup({
      lineup:lu,
      startSlot:1,
      pitcherMixture:[{weight:1,pitcher:p}],
      league,
      environmentalContext:null,
      weights:{batter:0.5,pitcher:0.5},
      random,
      playCalibration,
    });
    counts[r.nextSlot]+=1;
  }
  const denom=TRIALS+4.5;
  return Object.fromEntries(Array.from({length:9},(_,j)=>[String(j+1),(counts[j+1]+0.5)/denom]));
}
function loss(dist, observed){
  const p=Math.max(1e-12,Number(dist[String(observed)]||0));
  let brier=0;
  for(let s=1;s<=9;s++){
    const q=Number(dist[String(s)]||0);
    const y=s===observed?1:0;
    brier+=(q-y)**2;
  }
  const best=Object.entries(dist).sort((a,b)=>b[1]-a[1])[0][0];
  return {logloss:-Math.log(p),brier,observed_probability:p,top1:Number(best)===observed};
}

const rows=[];
for(const game of input.games){
  for(const side of ['top','bottom']){
    const observed=Number(game.observed[side==='top'?'top2_start_slot':'bottom2_start_slot']);
    if(!(observed>=1 && observed<=9)) continue;
    const leagueDist=distribution(game,side,'league');
    const playerDist=distribution(game,side,'player_asof');
    rows.push({
      gid:game.gid,date:game.date,side,observed_slot:observed,
      league:{...loss(leagueDist,observed),distribution:leagueDist},
      player_asof:{...loss(playerDist,observed),distribution:playerDist},
    });
  }
}
const payload={
  version:'i1-state-ab-evaluation-v1',
  generated_at:new Date().toISOString(),
  season:input.season,
  market_inputs_used:false,
  observed_i2_start_slot_used_as_predictor:false,
  trials_per_half:TRIALS,
  n_halves:rows.length,
  rows,
};
fs.mkdirSync(path.dirname(OUTPUT),{recursive:true});
fs.writeFileSync(OUTPUT,JSON.stringify(payload));
console.log(JSON.stringify({n_halves:rows.length,trials_per_half:TRIALS},null,2));
