import { assertBaseballOnly, validOrder } from './i2_source_governance.mjs';
export const ROTOWIRE_PUBLIC_URL='https://www.rotowire.com/baseball/daily-lineups.php';
function decode(s='') {
  return s.replace(/&#(x[0-9a-f]+|\d+);/gi,(_,n)=>String.fromCodePoint(n[0].toLowerCase()==='x'?parseInt(n.slice(1),16):Number(n)))
    .replace(/&quot;/g,'"').replace(/&apos;|&#39;/g,"'").replace(/&amp;/g,'&').replace(/&nbsp;/g,' ');
}
const text=s=>decode((s || '').replace(/<[^>]*>/g,' ')).replace(/\s+/g,' ').trim();
const attrs=s=>Object.fromEntries([...s.matchAll(/([\w-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')/g)].map(m=>[m[1],decode(m[2]??m[3])]));
const has=(tag,c)=>(attrs(tag).class || '').split(/\s+/).includes(c);
function elements(html,tag,cls) {
  return [...html.matchAll(new RegExp(`<${tag}\\b([^>]*)>`,'gi'))].filter(m=>has(m[1],cls)).flatMap(m=>{
    const start=m.index+m[0].length,end=html.indexOf(`</${tag}>`,start);
    return end<0?[]:[[m[0],m[1],html.slice(start,end)]];
  });
}
function easternTime(date,label) {
  const m=/^(\d{1,2}):(\d{2})\s*(AM|PM)\s*ET$/.exec(label);
  if (!m || Number(m[1])<1 || Number(m[1])>12 || Number(m[2])>59) return null;
  const hour=Number(m[1])%12+(m[3]==='PM'?12:0),base=Date.parse(`${date}T${String(hour).padStart(2,'0')}:${m[2]}:00Z`);
  const offset=new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',timeZoneName:'shortOffset'}).formatToParts(new Date(base)).find(p=>p.type==='timeZoneName').value.match(/GMT([+-]\d+)/);
  if (!offset) return null;
  return new Date(base-Number(offset[1])*3600000).toISOString();
}
function playerLink(html) {
  const matches=[...html.matchAll(/<a\b([^>]*)>([\s\S]*?)<\/a>/gi)].filter(m=>/^\/baseball\/player\//.test(attrs(m[1]).href || ''));
  if(matches.length!==1)return null;
  const name=attrs(matches[0][1]).title || text(matches[0][2]);
  // Never promote an abbreviated display name to an exact player identity.
  return /^[A-Z]\.\s/.test(name)?null:name;
}
// Strict selectors and explicit output allowlist: HTML/odds/scripts never leave this adapter.
export function parseRotowirePublic(html,date,retrievedAt=new Date().toISOString()) {
  const main=[...html.matchAll(/<main\b([^>]*)>/gi)].find(m=>attrs(m[1])['data-sportfull']==='baseball');
  if (!main || attrs(main[1])['data-gamedate']!==date) throw new Error('ROTOWIRE_PUBLIC_DATE_MISMATCH');
  const clean=html.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi,'').replace(/<!--[\s\S]*?-->/g,'');
  const starts=[...clean.matchAll(/<div\b([^>]*)>/gi)].filter(m=>has(m[1],'lineup') && has(m[1],'is-mlb')).map(m=>m.index);
  const rows=[];
  for(let i=0;i<starts.length;i++) {
    const card=clean.slice(starts[i],starts[i+1]??clean.length);
    const gameDate=easternTime(date,text(elements(card,'div','lineup__time')[0]?.[2]));
    const codes=elements(card,'div','lineup__abbr').map(m=>text(m[2]));
    if (!gameDate || codes.length!==2 || codes.some(c=>!/^\w{2,3}$/.test(c)))continue;
    const teams=['away','home'].map((side,j)=>{
      const lists=elements(card,'ul','lineup__list').filter(m=>has(m[1],side==='away'?'is-visit':'is-home'));
      if (lists.length!==1)return null;
      const list=lists[0][2],status=elements(list,'li','lineup__status')[0];
      const confirmed=Boolean(status && has(status[1],'is-confirmed') && text(status[2])==='Confirmed Lineup');
      const expected=Boolean(status && has(status[1],'is-expected') && text(status[2])==='Expected Lineup');
      const hitters=elements(list,'li','lineup__player');
      const players=hitters.map(m=>playerLink(m[2]));
      const highlights=elements(list,'li','lineup__player-highlight');
      const primary=highlights.find(m=>elements(m[2],'div','tag').some(t=>text(t[2])==='PRIM'));
      const starting=highlights.filter(m=>!elements(m[2],'div','tag').some(t=>text(t[2])==='PRIM'));
      const name=starting.length===1?playerLink(starting[0][2]):null;
      const meta={provider:'ROTOWIRE',source:ROTOWIRE_PUBLIC_URL,transport:'PUBLIC_HTML',timestamp:null,retrievedAt};
      return {side,code:codes[j],lineup:(confirmed || expected) && validOrder(players)?{...meta,confirmed,players,
        handedness:hitters.map((m,k)=>({name:players[k],bats:text(elements(m[2],'span','lineup__bats')[0]?.[2]) || null}))}:null,
        starter:name?{...meta,name,confirmed:false,role:'STARTER'}:null,primaryPitcher:primary?playerLink(primary[2]):null,opener:null};
    });
    if(teams.every(Boolean))rows.push({gameDate,teams});
  }
  if(!rows.length || !rows.some(r=>r.teams.some(t=>t.lineup || t.starter)))throw new Error('ROTOWIRE_PUBLIC_SCHEMA_UNAVAILABLE');
  assertBaseballOnly(rows);
  return rows;
}
export async function fetchRotowirePublic(date,fetchImpl=globalThis.fetch) {
  const url=new URL(ROTOWIRE_PUBLIC_URL);url.searchParams.set('date',date);
  const r=await fetchImpl(url,{signal:AbortSignal.timeout(15000),headers:{accept:'text/html','user-agent':'MLB-I2-Baseball-Inputs/1.0'}});
  if(!r.ok)throw new Error(`HTTP_${r.status}`);
  return parseRotowirePublic(await r.text(),date);
}
