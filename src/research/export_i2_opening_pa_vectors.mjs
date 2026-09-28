#!/usr/bin/env node
// Pregame-only PA vectors. Actual I2 slots and outcomes are joined after export.
import fs from 'node:fs';
import path from 'node:path';
import { predictI2EventVector } from '../model/i2_vnext_event_model.js';
import { applyEnvironmentalEventVector } from '../event_probability_engine.js';
import { venueForReplayGame } from './i2_vnext_replay_venue.mjs';
const arg=(key,otherwise)=>{const i=process.argv.indexOf(key);return i<0?otherwise:process.argv[i+1];};
const input=arg('--input','/workspace/scratch/e39cba13c44c/research_inputs/replay_2025_inputs.json');
const manifestPath=arg('--manifest','/workspace/scratch/e39cba13c44c/research_inputs/walkforward_2025/manifest.json');
const arsenalPath=arg('--arsenal','data/derived/i2_vnext/arsenal_profile_2024.json');
const parksPath=arg('--parks','/workspace/scratch/e39cba13c44c/research_inputs/park/savant_venue_profiles_2024_3yr.json');
const output=arg('--output','/workspace/scratch/e39cba13c44c/i2_opening_pa_vectors_2025.json');
const replay=JSON.parse(fs.readFileSync(input));
const manifest=JSON.parse(fs.readFileSync(manifestPath));
if(replay.market_inputs_used!==false||manifest.market_inputs_used!==false) throw new Error('Market input present');
const arsenal=JSON.parse(fs.readFileSync(arsenalPath));
const parks=JSON.parse(fs.readFileSync(parksPath));
const models=new Map(manifest.entries.map(e=>{const m=JSON.parse(fs.readFileSync(path.join(path.dirname(manifestPath),e.model_file)));if(m.market_inputs_used!==false)throw new Error('Model market input present');return [e.model_file,m];}));
const rows=[];
for(const g of replay.games){
 const date=String(g.date).replace(/[^0-9]/g,'').slice(0,8);
 const entry=manifest.entries.filter(e=>e.effective_from.replace(/-/g,'')<=date).at(-1);
 if(!entry)throw new Error(`Missing monthly model ${date}`);
 const model=models.get(entry.model_file);
 if(model.training.end.replace(/-/g,'')>=date)throw new Error(`Leakage ${date}`);
 const venue=venueForReplayGame(g,parks);
 for(const side of ['top','bottom']){
  const lineup=side==='top'?g.away_lineup:g.home_lineup;
  const pitcher=side==='top'?g.home_starter:g.away_starter;
  if(lineup.length!==9)throw new Error(`Invalid lineup ${g.gid}`);
  const entries=lineup.map((b,slot)=>{
   const bats=String(b.bats).toUpperCase(),throws=String(pitcher.throws).toUpperCase();
   const batterSide=bats==='B'?(throws==='L'?'R':'L'):(bats==='L'?'L':'R');
   const neutral=predictI2EventVector({batterId:Number(b.mlbam),pitcherId:Number(pitcher.mlbam),batterSide,pitcherThrows:throws,model,arsenalProfile:arsenal});
   const vector=applyEnvironmentalEventVector({neutralVector:neutral,environmentalContext:venue.profile,batterSide}).probabilities;
   if(Math.abs(Object.values(vector).reduce((a,v)=>a+v,0)-1)>1e-8)throw new Error('Invalid vector');
   return {slot:slot+1,batter:Number(b.mlbam),vector};
  });
  rows.push({gid:g.gid,date,side,pitcher:Number(pitcher.mlbam),venue_status:venue.status,model_file:entry.model_file,entries});
 }
}
fs.writeFileSync(output,JSON.stringify({version:'i2-opening-pa-vectors-2025-v1',market_inputs_used:false,pregame_only:true,rows}));
console.log(JSON.stringify({games:replay.games.length,halves:rows.length,output}));
