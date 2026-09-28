#!/usr/bin/env node
/** Pregame vectors for the first three guaranteed I1 batting-order slots. */
import fs from 'node:fs';
import path from 'node:path';
import {buildNeutralEventVector} from '../event_probability_engine.js';

function arg(name, fallback=null) {
  const i=process.argv.indexOf(name);
  return i>=0 ? process.argv[i+1] : fallback;
}
const INPUT=arg('--input');
const OUTPUT=arg('--output');
if (!INPUT || !OUTPUT) throw new Error('Provide --input and --output');
const input=JSON.parse(fs.readFileSync(INPUT,'utf8'));
if (input.market_inputs_used!==false || input.observed_i2_start_slot_used_as_predictor!==false) {
  throw new Error('Opening PA source is not market- and target-isolated');
}
if (input.season!==2024 || !input.league_event_rates) throw new Error('Expected governed 2024 I1 A/B inputs');

const rows=[];
for (const game of input.games) {
  for (const side of ['top','bottom']) {
    const lineup=side==='top' ? game.away_lineup : game.home_lineup;
    const pitcher=side==='top' ? game.home_starter : game.away_starter;
    if (lineup.length!==9 || !pitcher.i1_event_rates_asof) throw new Error(`Missing pregame inputs for ${game.gid}`);
    for (let i=0;i<3;i++) {
      rows.push({
        gid:game.gid,date:game.date,side,slot:i+1,
        batter_id:lineup[i].id,pitcher_id:pitcher.id,
        probabilities:buildNeutralEventVector({
          batter:lineup[i].i1_event_rates_asof,
          pitcher:pitcher.i1_event_rates_asof,
          league:input.league_event_rates,
          weights:{batter:0.5,pitcher:0.5},
        }),
      });
    }
  }
}
const out={
  version:'i1-opening-pa-pregame-vectors-v1',
  season:2024,
  market_inputs_used:false,
  observed_pa_outcomes_used_as_predictors:false,
  method:'existing 50/50 log-odds PA event formula; strictly pregame 2024 player-as-of rates; batting slots 1-3',
  n:rows.length,rows,
};
fs.mkdirSync(path.dirname(OUTPUT),{recursive:true});
fs.writeFileSync(OUTPUT,JSON.stringify(out));
console.log(JSON.stringify({n:out.n,games:input.games.length}));
