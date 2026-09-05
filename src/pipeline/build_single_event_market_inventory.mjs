import fs from 'node:fs/promises';
import path from 'node:path';

const API_BASE='https://api.the-odds-api.com';
const apiKey=String(process.env.ODDS_API_KEY||'').trim();
if(!apiKey) throw new Error('ODDS_API_KEY is required');

// Known MLB historical example from The Odds API docs: TB @ LAA, 2024-04-09.
const eventId='c163b5f5f4579c8293266956ccf3d9bd';
const date='2024-04-08T15:10:00Z';
const candidateMarkets=['totals_9th_inning','totals_9th','9th_inning_totals','totals_inning_9','h2h_9th_inning','spreads_9th_inning','totals_1st_5_innings'];
const out='docs/market_inventory/2026-08-30_lad_det_all_markets.md';
const rawOut='data/runtime/i2/2026-08-30_lad_det_all_markets_raw.json';

async function get(pathname,params={}) {
  const u=new URL(pathname,API_BASE);
  u.searchParams.set('apiKey',apiKey);
  for(const [k,v] of Object.entries(params)) if(v!==undefined&&v!==null&&v!=='') u.searchParams.set(k,String(v));
  const r=await fetch(u,{headers:{accept:'application/json','user-agent':'MLB-I9-Historical-Probe/1.0'}});
  const text=await r.text();
  let body; try { body=JSON.parse(text); } catch { body={raw:text}; }
  return {status:r.status,ok:r.ok,body,headers:{remaining:r.headers.get('x-requests-remaining'),used:r.headers.get('x-requests-used'),last:r.headers.get('x-requests-last')}};
}

const results={};
for(const market of candidateMarkets){
  results[market]=await get(`/v4/historical/sports/baseball_mlb/events/${eventId}/odds`,{date,bookmakers:'draftkings,fanduel',markets:market,oddsFormat:'american'});
}
const summary=Object.fromEntries(Object.entries(results).map(([market,r])=>[market,{status:r.status,ok:r.ok,error:r.body?.error_code||r.body?.message||null,books:r.body?.data?.bookmakers?.map(b=>({key:b.key,markets:b.markets?.map(m=>m.key)}))||[],quota:r.headers}]));
const payload={generatedAt:new Date().toISOString(),provider:'The Odds API v4',eventId,date,candidateMarkets,summary,results};
const md=['# The Odds API v4 historical MLB I9 probe','',`Generated: ${payload.generatedAt}`,'',`Historical event: ${eventId}`,'',`Snapshot requested: ${date}`,'','## Results','', '```json',JSON.stringify(summary,null,2),'```',''];
await fs.mkdir(path.dirname(out),{recursive:true});
await fs.mkdir(path.dirname(rawOut),{recursive:true});
await fs.writeFile(out,md.join('\n'));
await fs.writeFile(rawOut,JSON.stringify(payload,null,2)+'\n');
console.log(JSON.stringify(summary,null,2));
