import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { loadForecast, validateForecast, relevantRuns } from '../src/analysis/i2_forecast_availability.mjs';
const date=new Date().toISOString().slice(0,10);
const dir=fs.mkdtempSync(path.join(process.cwd(),'.i2-postmortem-test-'));
const file=path.join(dir,`${date}_frozen_predictions.json`);
const env={GITHUB_TOKEN:'test',GITHUB_REPOSITORY:'test/repo'};
const empty=async()=>({ok:true,json:async()=>({total_count:0,workflow_runs:[]})});
test.after(()=>fs.rmSync(dir,{recursive:true,force:true}));
test('scheduled no-forecast has no generated observations or file',async()=>{
 assert.equal(await loadForecast({date,file,mode:'auto',env,fetchImpl:empty}),null);
 assert.equal(fs.existsSync(file),false);
});
test('explicit missing forecast stays fatal',async()=>{
 await assert.rejects(loadForecast({date,file}),/Missing prediction/);
});
test('failed, successful, and active producer runs all require artifact',async()=>{
 for(const conclusion of ['failure','success',null]) {
 const fetchImpl=async()=>({ok:true,json:async()=>({total_count:1,workflow_runs:[{id:1,path:'.github/workflows/i2_daily_run.yml',head_branch:'main',conclusion}]})});
 await assert.rejects(loadForecast({date,file,mode:'auto',env,fetchImpl}),/Missing expected/);
 }
});
test('API errors cannot become no-forecast',async()=>{
 await assert.rejects(loadForecast({date,file,mode:'auto',env,fetchImpl:async()=>({ok:false,status:403})}),/HTTP 403/);
});
test('partial history is fatal',async()=>{
 await assert.rejects(loadForecast({date,file,mode:'auto',env,fetchImpl:async()=>({ok:true,json:async()=>({total_count:1,workflow_runs:[]})})}),/Incomplete/);
});
test('research and explicitly other-date runs do not establish production expectation',()=>{
 assert.equal(relevantRuns([{path:'.github/workflows/research.yml',head_branch:'main'},{path:'.github/workflows/i2_daily_run.yml',head_branch:'main',display_title:'I2 forecast date=2026-09-30'}],date).length,0);
});
test('existing production sibling is not an intentional no-forecast',async()=>{
 const sibling=path.join(dir,`${date}_v04_predictions.json`); fs.writeFileSync(sibling,'{}');
 try { await assert.rejects(loadForecast({date,file,mode:'auto',env,fetchImpl:empty}),/existing forecast evidence/); }
 finally { fs.unlinkSync(sibling); }
});
test('actual historical frozen forecast validates without network and is unchanged',async()=>{
 const historical='data/runtime/i2/2026-09-30_frozen_predictions.json';
 const before=fs.readFileSync(historical,'utf8');
 const pred=await loadForecast({date:'2026-09-30',file:historical,fetchImpl:()=>{throw Error('network');}});
 assert.equal(pred.games.length,4); assert.equal(fs.readFileSync(historical,'utf8'),before);
});
test('malformed JSON, date, games and probabilities fail closed',async()=>{
 for(const data of ['{',JSON.stringify({date,games:{}}),JSON.stringify({date:'2000-01-01',games:[]}),JSON.stringify({date,games:[{gamePk:1,modelStatus:'FROZEN_RESEARCH_PROJECTION',under05:null}]})]) {
 fs.writeFileSync(file,data);
 try { await assert.rejects(loadForecast({date,file,mode:'auto',env,fetchImpl:empty})); }
 finally { fs.unlinkSync(file); }
 }
 assert.throws(()=>validateForecast({},date));
});
test('season rollup independently preserves measured history, with no invented target date',()=>{
 const output=path.join(dir,'season.json');
 const run=spawnSync(process.execPath,['src/analysis/i2_rotowire_accuracy_season.mjs'],{env:{...process.env,I2_YEAR:'2026',I2_ROTOWIRE_ACCURACY_OUTPUT:output},encoding:'utf8'});
 assert.equal(run.status,0,run.stderr);
 const result=JSON.parse(fs.readFileSync(output));
 assert.ok(result.rows.length>0); assert.ok(!result.datesIncluded.includes('2026-10-15'));
});
test('malformed seasonal input is not silently omitted',()=>{
 const input=path.join(dir,'reports'); fs.mkdirSync(input); fs.writeFileSync(path.join(input,'2026-10-15_postmortem.json'),'{');
 const run=spawnSync(process.execPath,['src/analysis/i2_rotowire_accuracy_season.mjs'],{env:{...process.env,I2_YEAR:'2026',I2_POSTMORTEM_DIR:input,I2_ROTOWIRE_ACCURACY_OUTPUT:path.join(dir,'bad.json')}});
 assert.notEqual(run.status,0); assert.equal(fs.existsSync(path.join(dir,'bad.json')),false);
});
test('deleted persisted forecast is detected through git history',async()=>{
 const repo=path.join(dir,'repo'); fs.mkdirSync(repo);
 const git=(...args)=>{const r=spawnSync('git',args,{cwd:repo,encoding:'utf8'}); assert.equal(r.status,0,r.stderr);};
 git('init'); git('config','user.name','Test'); git('config','user.email','test@example.com');
 fs.mkdirSync(path.join(repo,'.github/workflows'),{recursive:true});
 for(const name of ['i2_daily_run.yml','i2_v04_today.yml']) fs.writeFileSync(path.join(repo,'.github/workflows',name),'on:\n  workflow_dispatch:\n');
 const name=`${date}_frozen_predictions.json`; fs.writeFileSync(path.join(repo,name),'{}');
 git('add','.');git('commit','-m','fixture');fs.unlinkSync(path.join(repo,name));
 const cwd=process.cwd();
 try {process.chdir(repo);await assert.rejects(loadForecast({date,file:name,mode:'auto',env,fetchImpl:empty}),/Missing persisted/);}
 finally {process.chdir(cwd);}
});
test('historical grading keeps probabilities; missing scores never become zero; feed failure preserves report',()=>{
 const historical='data/runtime/i2/2026-09-30_frozen_predictions.json';
 const original=fs.readFileSync(historical,'utf8');
 const mock=path.join(dir,'mock.mjs'), output=path.join(dir,'historical.json');
 for(const scenario of ['final','missing','failed']) {
 fs.writeFileSync(mock,`globalThis.fetch=async()=>({ok:${scenario!=='failed'},status:503,json:async()=>({gameData:{status:{abstractGameState:'Final'}},liveData:{linescore:{innings:[{num:2,away:${scenario==='missing'?'{}':'{runs:0}'},home:{runs:0}}]}}})});`);
 const before=fs.existsSync(output)?fs.readFileSync(output,'utf8'):null;
 const run=spawnSync(process.execPath,['--import',mock,'src/analysis/i2_postmortem_today.mjs'],{env:{...process.env,I2_DATE:'2026-09-30',I2_PREDICTIONS:historical,I2_POSTMORTEM_OUTPUT:output},encoding:'utf8'});
 if(scenario==='failed'){ assert.notEqual(run.status,0);assert.equal(fs.readFileSync(output,'utf8'),before); }
 else {
 assert.equal(run.status,0,run.stderr); const report=JSON.parse(fs.readFileSync(output));
 assert.equal(report.summary.gradedGames,scenario==='final'?4:0);
 assert.equal(report.summary.modelBrier===null,scenario==='missing');
 if(scenario==='final') {
 const pred=JSON.parse(original); assert.equal(report.rows[0].brier,(pred.games[0].under05-1)**2);
 }
 }
 }
 assert.equal(fs.readFileSync(historical,'utf8'),original);
});
