import fs from 'node:fs';
import path from 'node:path';

const YEAR=String(process.env.I2_YEAR || new Date().getUTCFullYear());
const INPUT_DIR=process.env.I2_POSTMORTEM_DIR || 'data/runtime/i2';
const OUTPUT=process.env.I2_ROTOWIRE_ACCURACY_OUTPUT || `data/derived/i2/rotowire_provisional_accuracy_${YEAR}.json`;

const mean = values => {
  const rows=values.filter(Number.isFinite);
  return rows.length ? rows.reduce((a,b)=>a+b,0)/rows.length : null;
};
function summarize(rows){
  return {
    sidesGraded:rows.length,
    gamesRepresented:new Set(rows.map(r=>String(r.gamePk))).size,
    exactLineups:rows.filter(r=>r.exactLineup).length,
    exactLineupRate:rows.length?rows.filter(r=>r.exactLineup).length/rows.length:null,
    exactPlayerSets:rows.filter(r=>r.exactPlayerSet).length,
    exactPlayerSetRate:rows.length?rows.filter(r=>r.exactPlayerSet).length/rows.length:null,
    meanPlayerSetAccuracy:mean(rows.map(r=>r.playerSetAccuracy)),
    meanSlotAccuracy:mean(rows.map(r=>r.slotAccuracy)),
    meanTop4SlotAccuracy:mean(rows.map(r=>r.top4SlotAccuracy)),
    meanObservationLeadHours:mean(rows.map(r=>r.observationLeadHours))
  };
}

const files=fs.existsSync(INPUT_DIR)
  ? fs.readdirSync(INPUT_DIR).filter(f=>new RegExp(`^${YEAR}-\\d{2}-\\d{2}_postmortem\\.json$`).test(f)).sort()
  : [];

const rows=[];
for(const file of files){
  let payload;
  try { payload=JSON.parse(fs.readFileSync(path.join(INPUT_DIR,file),'utf8')); } catch { continue; }
  const date=file.slice(0,10);
  for(const game of payload.rows || []){
    for(const side of ['away','home']){
      const a=game.rotowireProvisionalAudit?.[side];
      if(!a || a.confirmed || !a.accuracy) continue;
      rows.push({
        date,
        month:date.slice(0,7),
        gamePk:game.gamePk,
        matchup:game.matchup,
        side,
        source:a.source||null,
        retrievedAt:a.retrievedAt||null,
        observationLeadHours:Number.isFinite(a.observationLeadHours)?a.observationLeadHours:null,
        playersMatched:a.accuracy.playersMatched,
        playerSetAccuracy:a.accuracy.playerSetAccuracy,
        exactPlayerSet:Boolean(a.accuracy.exactPlayerSet),
        exactSlots:a.accuracy.exactSlots,
        slotAccuracy:a.accuracy.slotAccuracy,
        top4ExactSlots:a.accuracy.top4ExactSlots,
        top4SlotAccuracy:a.accuracy.top4SlotAccuracy,
        exactLineup:Boolean(a.accuracy.exactLineup)
      });
    }
  }
}

const months={};
for(const month of [...new Set(rows.map(r=>r.month))].sort()) months[month]=summarize(rows.filter(r=>r.month===month));

const out={
  year:YEAR,
  generatedAt:new Date().toISOString(),
  source:'RotoWire provisional lineups vs eventual MLB starting lineups',
  sourceRole:'AUDIT_ONLY',
  affectsPrediction:false,
  affectsEligibility:false,
  methodology:'Identity-aware comparison using MLB-equivalent names/IDs where available; no score is fed back into production.',
  datesIncluded:[...new Set(rows.map(r=>r.date))].sort(),
  ...summarize(rows),
  monthly:months,
  rows
};
fs.mkdirSync(path.dirname(OUTPUT),{recursive:true});
fs.writeFileSync(OUTPUT,JSON.stringify(out,null,2));
console.log(JSON.stringify({...out,rows:undefined},null,2));
