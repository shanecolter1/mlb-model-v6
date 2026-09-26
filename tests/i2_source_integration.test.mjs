import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
const fixture=JSON.parse(fs.readFileSync('tests/fixtures/i2_source_parity.json','utf8'));
function run(mode) {
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'i2-sourcing-')),output=path.join(dir,'predictions.json');
 try {
  execFileSync(process.execPath,['--import','./tests/fixtures/i2_baseball_fetch.mjs','src/pipeline/run_i2_today.mjs'],{env:{...process.env,I2_DATE:fixture.date,I2_CUTOFF:fixture.date+'T00:00:00Z',I2_TRIALS:String(fixture.trials),I2_OUTPUT:output,ROTOWIRE_API_KEY:mode==='provisional'?'fixture-only':'',I2_TEST_SOURCE_MODE:mode,I2_ROSTERRESOURCE_SNAPSHOT:'',I2_BASEBALL_REPORTS:''},stdio:'pipe'});
  return JSON.parse(fs.readFileSync(output,'utf8'));
 } finally {fs.rmSync(dir,{recursive:true,force:true});}
}
test('identical MLB inputs produce exact pre-upgrade production/calibration probabilities',()=>{
 const g=run('confirmed').games[0];assert.equal(g.modelStatus,'FROZEN_RESEARCH_PROJECTION');
 for(const [key,value] of Object.entries(fixture.probabilities))assert.deepEqual(g[key],value,key);
 assert.equal(g.bettingEligibility.eligible,true);assert.equal(g.inputAudit.away.lineup.status,'CONFIRMED_MLB');
});
test('expected RotoWire feeds reach real production runner; source confidence does not change probabilities',()=>{
 const g=run('provisional').games[0];assert.equal(g.modelStatus,'FROZEN_RESEARCH_PROJECTION');
 assert.equal(g.inputAudit.away.lineup.status,'PROVISIONAL_ROTOWIRE');assert.equal(g.lineupConfirmed,false);assert.equal(g.inputAudit.away.lineup.confidence,'LOW');
 for(const [key,value] of Object.entries(fixture.probabilities))assert.deepEqual(g[key],value,key);
});
test('starter changes during simulation invalidate freeze and suppress ranking',()=>{
 const p=run('change'),g=p.games[0];assert.equal(g.modelStatus,'PROJECTION_INVALIDATED');assert.equal(g.bettingEligibility.eligible,false);assert.ok(g.bettingEligibility.reasons.includes('PROJECTION_INVALIDATED_STARTER_CHANGE'));assert.equal(p.ranking.length,0);
});
