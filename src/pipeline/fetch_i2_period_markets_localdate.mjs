import fs from 'node:fs';

const date=process.env.I2_DATE||new Date().toISOString().slice(0,10);
const key=process.env.ODDS_API_KEY;
if(!key)throw new Error('ODDS_API_KEY missing');
async function get(path,params={}){const u=new URL(path,'https://api.theoddsapi.com');for(const[k,v]of Object.entries(params))u.searchParams.set(k,String(v));const r=await fetch(u,{headers:{'x-api-key':key,accept:'application/json'}});const t=await r.text();let b;try{b=JSON.parse(t)}catch{b={raw:t}};if(!r.ok)throw new Error(`${r.status} ${t.slice(0,400)}`);return b;}
// Central Daylight Time local-date window for Sep 8, 2026.
const from=`${date}T05:00:00Z`;
const next=new Date(`${date}T00:00:00Z`);next.setUTCDate(next.getUTCDate()+1);
const to=`${next.toISOString().slice(0,10)}T05:00:00Z`;
const ev=await get('/odds/',{sport_key:'baseball_mlb',markets:'h2h',commenceTimeFrom:from,commenceTimeTo:to});
const events=Array.isArray(ev?.data)?ev.data:[],probes=[];
for(const e of events){try{const p=await get('/period-markets/',{sport_key:'baseball_mlb',event_id:e.event_id});probes.push({event_id:e.event_id,away_team:e.away_team,home_team:e.home_team,start_time:e.start_time,response:p});}catch(error){probes.push({event_id:e.event_id,away_team:e.away_team,home_team:e.home_team,start_time:e.start_time,error:String(error.message||error)});}}
const rows=[];
for(const p of probes){const raw=p.response?.data,items=Array.isArray(raw)?raw:(raw&&typeof raw==='object'?[raw]:[]);for(const item of items)for(const m of item.markets||[])for(const b of m.books||[])rows.push({event_id:item.event_id||p.event_id,matchup:`${item.away_team||p.away_team} @ ${item.home_team||p.home_team}`,start_time:item.start_time||p.start_time,market:m.market,sportsbook:b.book,updated_at:b.updated_at||null,outcomes:b.outcomes||[]});}
const out={date,generatedAt:new Date().toISOString(),window:{from,to},events:events.length,rows,probes};
fs.writeFileSync(`data/runtime/i2/${date}_official_period_markets.json`,JSON.stringify(out,null,2)+'\n');
console.log(JSON.stringify({events:events.length,rows:rows.length,markets:[...new Set(rows.map(x=>x.market))],books:[...new Set(rows.map(x=>x.sportsbook))]},null,2));
