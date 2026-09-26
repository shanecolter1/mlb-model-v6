#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { buildNeutralEventVector } from '../event_probability_engine.js';

function arg(name, fallback=null) {
  const i=process.argv.indexOf(name);
  return i>=0 ? process.argv[i+1] : fallback;
}
const INPUT=arg('--input','data/derived/i2_vnext/i1_state_ab_2024_inputs.json');
const PLAY=arg('--play-calibration','data/derived/model_calibration/seasonal/production_pa_transition_table_shrunk.json');
const OUTPUT=arg('--output','data/derived/i2_vnext/i1_state_ab_2024_rows.json');
const MAX_PA=Number(arg('--max-pa','40'));

const input=JSON.parse(fs.readFileSync(INPUT,'utf8'));
const raw=JSON.parse(fs.readFileSync(PLAY,'utf8'));
const league=input.league_event_rates;
if (!raw?.states) throw new Error('Expected seasonal transition artifact with states');

const EVENT_TO_TRANSITION={
  strikeout:'out',
  ball_in_play_out:'out',
  walk:'bb',
  hit_by_pitch:'bb',
  single:'single',
  double:'double',
  triple:'triple',
  home_run:'hr',
};

function rates(item, mode){
  return mode==='player_asof' ? item.i1_event_rates_asof : league;
}

function eventVector(batter,pitcher,mode){
  return buildNeutralEventVector({
    batter:rates(batter,mode),
    pitcher:rates(pitcher,mode),
    league,
    weights:{batter:0.5,pitcher:0.5},
  });
}

function stateKey(outs,mask,slot){ return `${outs}|${mask}|${slot}`; }

function exactDistribution(game,side,mode){
  const isTop=side==='top';
  const lineup=isTop?game.away_lineup:game.home_lineup;
  const pitcher=isTop?game.home_starter:game.away_starter;
  let active=new Map([[stateKey(0,0,1),1]]);
  const absorbed=Array(10).fill(0);

  for(let pa=0;pa<MAX_PA && active.size;pa++){
    const next=new Map();
    for(const [key,stateProb] of active.entries()){
      const [outsText,maskText,slotText]=key.split('|');
      const outs=Number(outsText), mask=Number(maskText), slot=Number(slotText);
      const batter=lineup[slot-1];
      const vector=eventVector(batter,pitcher,mode);
      const nextSlot=slot===9?1:slot+1;

      for(const [event,eventProb] of Object.entries(vector)){
        if (!(eventProb>0)) continue;
        const source=EVENT_TO_TRANSITION[event];
        const transitions=raw.states[`${source}|${outs}|${mask}`];
        if (!Array.isArray(transitions) || !transitions.length) {
          throw new Error(`Missing transition state ${source}|${outs}|${mask}`);
        }
        for(const t of transitions){
          const p=stateProb*eventProb*Number(t.p||0);
          if (!(p>0)) continue;
          const newOuts=Math.min(3,outs+Math.max(0,Number(t.outs_added||0)));
          if(newOuts>=3){
            absorbed[nextSlot]+=p;
          } else {
            const nk=stateKey(newOuts,Number(t.post_mask||0),nextSlot);
            next.set(nk,(next.get(nk)||0)+p);
          }
        }
      }
    }
    active=next;
  }

  const tail=[...active.values()].reduce((a,b)=>a+b,0);
  if(tail>1e-8) throw new Error(`Exact I1 state tail too large after ${MAX_PA} PA: ${tail}`);
  const total=absorbed.reduce((a,b)=>a+b,0);
  if(!(total>0.999999 && total<=1.000001)) throw new Error(`I1 slot distribution mass=${total}`);
  return Object.fromEntries(Array.from({length:9},(_,j)=>[String(j+1),absorbed[j+1]/total]));
}

function loss(dist, observed){
  const p=Math.max(1e-15,Number(dist[String(observed)]||0));
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
    const leagueDist=exactDistribution(game,side,'league');
    const playerDist=exactDistribution(game,side,'player_asof');
    rows.push({
      gid:game.gid,date:game.date,side,observed_slot:observed,
      league:{...loss(leagueDist,observed),distribution:leagueDist},
      player_asof:{...loss(playerDist,observed),distribution:playerDist},
    });
  }
}
const payload={
  version:'i1-state-ab-evaluation-v2-exact',
  generated_at:new Date().toISOString(),
  season:input.season,
  market_inputs_used:false,
  observed_i2_start_slot_used_as_predictor:false,
  evaluation_method:'exact dynamic propagation through validated empirical event/base-out transition table',
  max_pa:MAX_PA,
  n_halves:rows.length,
  rows,
};
fs.mkdirSync(path.dirname(OUTPUT),{recursive:true});
fs.writeFileSync(OUTPUT,JSON.stringify(payload));
console.log(JSON.stringify({n_halves:rows.length,method:payload.evaluation_method},null,2));
