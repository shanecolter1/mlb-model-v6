import fs from 'node:fs/promises';
import path from 'node:path';

const API_BASE='https://api.theoddsapi.com';
const apiKey=String(process.env.ODDS_API_KEY||'').trim();
if(!apiKey) throw new Error('ODDS_API_KEY is required');
const eventId=process.env.EVENT_ID||'369e8c1f02400fd5e4528e7397bdcde6';
const out=process.env.OUT||'docs/market_inventory/2026-08-30_lad_det_all_markets.md';
const rawOut=process.env.RAW_OUT||'data/runtime/i2/2026-08-30_lad_det_all_markets_raw.json';

async function get(endpoint,params){const u=new URL(endpoint,API_BASE);for(const[k,v]of Object.entries(params||{}))if(v!==undefined&&v!==null&&v!=='')u.searchParams.set(k,String(v));const r=await fetch(u,{headers:{accept:'application/json','x-api-key':apiKey,'user-agent':'MLB-I2-Single-Event-Market-Inventory/1.2'}});const t=await r.text();let b;try{b=JSON.parse(t)}catch{b={raw:t}};return {status:r.status,ok:r.ok,body:b,url:u.pathname+'?'+u.searchParams.toString()};}

const core=await get('/odds/',{sport_key:'baseball_mlb',event_id:eventId});
const period=await get('/period-markets/',{sport_key:'baseball_mlb',event_id:eventId});
const props=await get('/props/',{sport_key:'baseball_mlb',event_id:eventId});
const endpoints={core,period,props};

function dataObjects(res){const d=res?.body?.data;return Array.isArray(d)?d:(d&&typeof d==='object'?[d]:[])}
const rows=[];
function pushBookRows(category, marketKey, books, eventObj={}){
  for(const b of books||[]) rows.push({category,market:marketKey||b.market,sportsbook:b.book||b.bookmaker||b.key,outcomes:b.outcomes||[],updated_at:b.updated_at||b.last_update,event:eventObj});
}
for(const [category,res] of Object.entries(endpoints)){
  for(const obj of dataObjects(res)){
    if(category==='core'){
      pushBookRows(category,null,obj.books,obj);
      continue;
    }
    if(Array.isArray(obj.markets)){
      for(const m of obj.markets) pushBookRows(category,m.market||m.key,m.books||m.bookmakers,obj);
      continue;
    }
    if(obj.market && (Array.isArray(obj.books)||Array.isArray(obj.bookmakers))){
      pushBookRows(category,obj.market,obj.books||obj.bookmakers,obj);
      continue;
    }
    if(obj.book && obj.market){
      rows.push({category,market:obj.market,sportsbook:obj.book,outcomes:obj.outcomes||[],updated_at:obj.updated_at,event:obj});
      continue;
    }
    if(obj.sportsbook && obj.market){
      rows.push({category,market:obj.market,sportsbook:obj.sportsbook,outcomes:obj.outcomes||[],updated_at:obj.updated_at,event:obj});
    }
  }
}
const rowsClean=rows.filter(r=>r.market&&r.sportsbook);
const markets=[...new Set(rowsClean.map(r=>r.market))].sort();
const books=[...new Set(rowsClean.map(r=>r.sportsbook))].sort();
const eventSource=rowsClean[0]?.event||{};
const matchup=eventSource.away_team&&eventSource.home_team?`${eventSource.away_team} @ ${eventSource.home_team}`:'Los Angeles Dodgers @ Detroit Tigers';
const esc=s=>String(s??'').replaceAll('|','\\|');
const md=[];
md.push(`# Single-event complete MLB market inventory — ${matchup}`,'',`Generated: ${new Date().toISOString()}`,'',`Event ID: \`${eventId}\``,'',`Endpoints queried: \`/odds/\`, \`/period-markets/\`, \`/props/\`. No market whitelist was applied to period or props calls; each endpoint was asked for all available markets for this event.`,'');
md.push('## Endpoint status','', '| Category | Endpoint | HTTP | Returned normalized rows | Raw data type | Raw data count |','|---|---|---:|---:|---|---:|');
for(const [c,r] of Object.entries(endpoints)){const d=r.body?.data;const rawType=Array.isArray(d)?'array':(d===null?'null':typeof d);const rawCount=Array.isArray(d)?d.length:(d&&typeof d==='object'?1:0);md.push(`| ${c} | ${esc(r.url)} | ${r.status} | ${rowsClean.filter(x=>x.category===c).length} | ${rawType} | ${rawCount} |`)}
md.push('','## Distinct market keys','');
for(const m of markets) md.push(`- \`${m}\``);
md.push('','## Sportsbook × market availability','');
md.push(`| Sportsbook | ${markets.map(esc).join(' | ')} |`);md.push(`|---|${markets.map(()=> '---:').join('|')}|`);
for(const b of books) md.push(`| ${esc(b)} | ${markets.map(m=>rowsClean.some(r=>r.sportsbook===b&&r.market===m)?'✓':'').join(' | ')} |`);
md.push('','## All returned bets by sportsbook','');
for(const b of books){md.push(`### ${b}`,'','| Category | Market | Outcomes | Updated |','|---|---|---|---|');for(const r of rowsClean.filter(x=>x.sportsbook===b).sort((a,z)=>a.category.localeCompare(z.category)||a.market.localeCompare(z.market))){const o=r.outcomes.map(x=>[x.description||x.name,x.point!==undefined?`line ${x.point}`:null,x.price!==undefined?`price ${x.price}`:null].filter(Boolean).join(' ')).join('; ');md.push(`| ${r.category} | ${esc(r.market)} | ${esc(o)} | ${esc(r.updated_at||'')} |`)}md.push('')}
await fs.mkdir(path.dirname(out),{recursive:true});await fs.mkdir(path.dirname(rawOut),{recursive:true});
await fs.writeFile(out,md.join('\n')+'\n');await fs.writeFile(rawOut,JSON.stringify({generatedAt:new Date().toISOString(),eventId,matchup,endpoints,normalized:{markets,books,rows:rowsClean}},null,2)+'\n');
console.log(JSON.stringify({eventId,matchup,markets,books,rowCount:rowsClean.length,categoryCounts:Object.fromEntries(Object.keys(endpoints).map(c=>[c,rowsClean.filter(r=>r.category===c).length])),rawShapes:Object.fromEntries(Object.entries(endpoints).map(([c,r])=>{const d=r.body?.data;return [c,{type:Array.isArray(d)?'array':typeof d,count:Array.isArray(d)?d.length:(d&&typeof d==='object'?1:0)}]})),out,rawOut},null,2));
