import fs from 'node:fs/promises';
import path from 'node:path';

const API_BASE='https://api.theoddsapi.com';
const apiKey=String(process.env.ODDS_API_KEY||'').trim();
if(!apiKey) throw new Error('ODDS_API_KEY is required');
const eventId=process.env.EVENT_ID||'799823f7421c5a5c018435aa85a7b250';
const out=process.env.OUT||'docs/market_inventory/2026-08-30_hou_nym_all_markets.md';
const rawOut=process.env.RAW_OUT||'data/runtime/i2/2026-08-30_hou_nym_all_markets_raw.json';

async function get(endpoint,params){const u=new URL(endpoint,API_BASE);for(const[k,v]of Object.entries(params||{}))if(v!==undefined&&v!==null&&v!=='')u.searchParams.set(k,String(v));const r=await fetch(u,{headers:{accept:'application/json','x-api-key':apiKey,'user-agent':'MLB-I2-Single-Event-Market-Inventory/1.0'}});const t=await r.text();let b;try{b=JSON.parse(t)}catch{b={raw:t}};return {status:r.status,ok:r.ok,body:b,url:u.pathname+'?'+u.searchParams.toString()};}

const core=await get('/odds/',{sport_key:'baseball_mlb',event_id:eventId});
const period=await get('/period-markets/',{sport_key:'baseball_mlb',event_id:eventId});
const props=await get('/props/',{sport_key:'baseball_mlb',event_id:eventId});
const endpoints={core,period,props};

function eventObjects(x){const d=x?.body?.data;return Array.isArray(d)?d:(d&&typeof d==='object'?[d]:[])}
const rows=[];
for(const [category,res] of Object.entries(endpoints)){
  for(const ev of eventObjects(res)){
    if(category==='core'){
      for(const b of ev.books||[]) rows.push({category,market:b.market,sportsbook:b.book,outcomes:b.outcomes||[],updated_at:b.updated_at,event:ev});
    } else {
      for(const m of ev.markets||[]) for(const b of m.books||[]) rows.push({category,market:m.market,sportsbook:b.book,outcomes:b.outcomes||[],updated_at:b.updated_at,event:ev});
    }
  }
}
const markets=[...new Set(rows.map(r=>r.market))].sort();
const books=[...new Set(rows.map(r=>r.sportsbook))].sort();
const matchup=rows[0]?`${rows[0].event.away_team} @ ${rows[0].event.home_team}`:eventId;
const esc=s=>String(s??'').replaceAll('|','\\|');
const md=[];
md.push(`# Single-event complete MLB market inventory — ${matchup}`,'',`Generated: ${new Date().toISOString()}`,'',`Event ID: \`${eventId}\``,'',`Endpoints queried: \`/odds/\`, \`/period-markets/\`, \`/props/\`. No market whitelist was applied to period or props calls; each endpoint was asked for all available markets for this event.`,'');
md.push('## Endpoint status','', '| Category | Endpoint | HTTP | Returned rows |','|---|---|---:|---:|');
for(const [c,r] of Object.entries(endpoints)) md.push(`| ${c} | ${esc(r.url)} | ${r.status} | ${rows.filter(x=>x.category===c).length} |`);
md.push('','## Distinct market keys','');
for(const m of markets) md.push(`- \`${m}\``);
md.push('','## Sportsbook × market availability','');
md.push(`| Sportsbook | ${markets.map(esc).join(' | ')} |`);md.push(`|---|${markets.map(()=> '---:').join('|')}|`);
for(const b of books) md.push(`| ${esc(b)} | ${markets.map(m=>rows.some(r=>r.sportsbook===b&&r.market===m)?'✓':'').join(' | ')} |`);
md.push('','## All returned bets by sportsbook','');
for(const b of books){md.push(`### ${b}`,'','| Category | Market | Outcomes | Updated |','|---|---|---|---|');for(const r of rows.filter(x=>x.sportsbook===b).sort((a,z)=>a.category.localeCompare(z.category)||a.market.localeCompare(z.market))){const o=r.outcomes.map(x=>[x.description||x.name,x.point!==undefined?`line ${x.point}`:null,x.price!==undefined?`price ${x.price}`:null].filter(Boolean).join(' ')).join('; ');md.push(`| ${r.category} | ${esc(r.market)} | ${esc(o)} | ${esc(r.updated_at||'')} |`)}md.push('')}
await fs.mkdir(path.dirname(out),{recursive:true});await fs.mkdir(path.dirname(rawOut),{recursive:true});
await fs.writeFile(out,md.join('\n')+'\n');await fs.writeFile(rawOut,JSON.stringify({generatedAt:new Date().toISOString(),eventId,matchup,endpoints,normalized:{markets,books,rows}},null,2)+'\n');
console.log(JSON.stringify({eventId,matchup,markets,books,rowCount:rows.length,out,rawOut},null,2));
