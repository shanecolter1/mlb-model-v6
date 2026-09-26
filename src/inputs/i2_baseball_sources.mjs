import fs from 'node:fs';
import { fetchRotowirePublic } from './rotowire_public.mjs';
import { assertBaseballOnly, norm, fresh, selectLineup, selectStarter, projectionGate, srmReview } from './i2_source_governance.mjs';

const MLB = 'https://statsapi.mlb.com';
const fullName = p => [p?.FirstName,p?.LastName].filter(Boolean).join(' ');
const TEAM_CODES = {108:['LAA'],109:['ARI','AZ'],110:['BAL'],111:['BOS'],112:['CHC'],113:['CIN'],114:['CLE'],115:['COL'],116:['DET'],117:['HOU'],118:['KC','KCR'],119:['LAD'],120:['WSH','WAS'],121:['NYM'],133:['ATH','OAK'],134:['PIT'],135:['SD','SDP'],136:['SEA'],137:['SF','SFG'],138:['STL'],139:['TB','TBR'],140:['TEX'],141:['TOR'],142:['MIN'],143:['PHI'],144:['ATL'],145:['CWS','CHW'],146:['MIA'],147:['NYY'],158:['MIL']};
const stamp = () => new Date().toISOString();
export async function safeSource(source, fn) {
  try { return { source, status: 'AVAILABLE', retrievedAt: stamp(), value: await fn() }; }
  catch (e) { return { source, status: `${source}_UNAVAILABLE`, retrievedAt: stamp(), reason: e?.message?.startsWith('HTTP_') ? e.message : 'MISSING_CREDENTIAL_OR_ACCESS_OR_INVALID_SCHEMA', value: null }; }
}
async function request(url, json = true) {
  const r = await fetch(url, {signal:AbortSignal.timeout(15000),headers:{accept:json?'application/json':'application/rss+xml', 'user-agent':'MLB-I2-Baseball-Inputs/1.0'}});
  if (!r.ok) throw new Error(`HTTP_${r.status}`);
  return json ? r.json() : r.text();
}
export function normalizeRotowire(payload, date, retrievedAt) {
  if (payload?.Date !== date || !Array.isArray(payload.Games)) throw new Error('INVALID_ROTOWIRE_SCHEMA_OR_DATE');
  return payload.Games.map(g => ({ gameDate: g.DateTime, teams: (g.Teams || []).map(t => ({
    code: t.Code, name: t.Name, side: Number(t.IsHome) === 1 ? 'home' : 'away',
    lineup: Array.isArray(t.Players) ? {provider:'ROTOWIRE',source:'RotoWire Projected Lineups',timestamp:t.UpdatedAt || null,retrievedAt,
      confirmed:['Confirmed','CONFIRMED','C'].includes(t.LineupStatus),
      players:[...t.Players].sort((a,b)=>Number(a.BattingSpot)-Number(b.BattingSpot)).filter(p=>Number(p.BattingSpot)>=1 && Number(p.BattingSpot)<=9).map(fullName),
      handedness:t.Players.map(p=>({name:fullName(p),bats:p.Bats || p.BatSide || null}))} : null,
    starter:t.StartingPitcher ? {provider:'ROTOWIRE',source:'RotoWire Projected Starters',name:fullName(t.StartingPitcher),timestamp:t.UpdatedAt || null,retrievedAt,confirmed:false,
      role:t.OpenerPitcher ? 'OPENER' : 'STARTER'} : null,
    opener:t.OpenerPitcher ? fullName(t.OpenerPitcher) : null,
  })) }));
}
function readSnapshot(file, date) {
  if (!file || !fs.existsSync(file)) throw new Error('NO_SNAPSHOT');
  const x = JSON.parse(fs.readFileSync(file,'utf8'));
  assertBaseballOnly(x);
  if (x.date !== date || !x.games || !fresh({timestamp:x.generatedAt})) throw new Error('STALE_OR_INVALID_SNAPSHOT');
  return x;
}
function xmlText(value = '') { return value.replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g,'$1').replace(/<[^>]*>/g,' ').replace(/&amp;/g,'&').replace(/&#39;/g,"'").trim(); }
export function parseNews(xml, now = Date.now()) {
  return [...xml.matchAll(/<item>([\s\S]*?)<\/item>/g)].flatMap(([,item]) => {
    const field = k => xmlText(item.match(new RegExp(`<${k}[^>]*>([\\s\\S]*?)<\\/${k}>`))?.[1]);
    const title=field('title'), timestamp=field('pubDate'), reason=title+' '+field('description');
    if (!fresh({timestamp},now,36)) return [];
    const workload=/opener|bullpen game|innings limit|pitch.count|rehab|Tommy John|abbreviated start|velocity|injury exit|return.*(IL|injur)/i.test(reason);
    const starter=/rotation|starter|starting pitcher|shutdown|shut down/i.test(reason);
    const lineup=/scratch|rest|day off|lineup|platoon|activat|call.?up|promot/i.test(reason);
    if (!workload && !starter && !lineup) return [];
    return [{source:field('link'),timestamp,retrievedAt:stamp(),reason,action:'REVIEW_NEWS',verified:false,explicit:false,
      recommendedAction:workload?'SRM_REVIEW_RECOMMENDED':starter?'STARTER_UPDATE_REQUIRED':'LINEUP_UPDATE_RECOMMENDED'}];
  });
}
export async function collectSources(date, env = process.env) {
  const rw = async endpoint => {
    if (!env.ROTOWIRE_API_KEY) throw new Error('MISSING_KEY');
    const u = new URL(`https://api.rotowire.com/Baseball/MLB/${endpoint}.php`);
    u.searchParams.set('key',env.ROTOWIRE_API_KEY);u.searchParams.set('date',date);u.searchParams.set('format','json');
    return normalizeRotowire(await request(u),date,stamp());
  };
  const publicPage=safeSource('ROTOWIRE_PUBLIC',()=>fetchRotowirePublic(date));
  const rotowire=async(endpoint,kind)=>{
    const page=await publicPage;
    if(page.value?.some(g=>g.teams.some(t=>t[kind]))) return page.value;
    return rw(endpoint);
  };
  const [lineups,starters,rr,reports,news] = await Promise.all([
    safeSource('ROTOWIRE',()=>rotowire('ProjectedLineups','lineup')),safeSource('ROTOWIRE_STARTERS',()=>rotowire('ProjectedStarters','starter')),
    // FanGraphs has no verified public JSON contract. Reviewed same-day export is explicit,
    // never reinterpret a depth chart as a game-confirmed lineup.
    safeSource('ROSTERRESOURCE',()=>readSnapshot(env.I2_ROSTERRESOURCE_SNAPSHOT,date)),
    safeSource('TEAM_BEAT_REPORTS',()=>readSnapshot(env.I2_BASEBALL_REPORTS || 'config/i2_baseball_reports.json',date)),
    safeSource('MLB_NEWS',async()=>parseNews(await request('https://www.mlb.com/feeds/news/rss.xml',false)))
  ]);
  const {value,...publicAudit}=await publicPage;
  lineups.publicPage=publicAudit;starters.publicPage=publicAudit;
  return {date,lineups,starters,rr,reports,news};
}
export function feedOrder(feed, side) {
  return Object.values(feed?.liveData?.boxscore?.teams?.[side]?.players || {}).filter(p=>Number(p.battingOrder)>0 && Number(p.battingOrder)%100===0).sort((a,b)=>Number(a.battingOrder)-Number(b.battingOrder)).map(p=>p.person?.fullName).filter(Boolean).slice(0,9);
}
function matchedRW(rows, game, side) {
  // Match both teams and time. A same-day doubleheader must never match by team alone.
  const matches=(rows || []).filter(g=>Math.abs(Date.parse(g.gameDate)-Date.parse(game.gameDate))<=30*60000 && ['away','home'].every(s=>{
    const team=game.teams?.[s]?.team;
    return g.teams.some(t=>t.side===s && (norm(t.name)===norm(team?.name) || (team?.abbreviation && norm(t.code)===norm(team.abbreviation)) || TEAM_CODES[team?.id]?.includes(t.code) || (['ATH','OAK'].includes(t.code) && team?.id===133)));
  }));
  return matches.length===1 ? matches[0].teams.find(t=>t.side===side) : null;
}
const previousCache=new Map();
async function previousOrder(game,side,date) {
  const id=game.teams[side].team.id, key=`${date}:${id}`;
  if (!previousCache.has(key)) previousCache.set(key,(async()=>{
    const end=new Date(`${date}T12:00:00Z`);end.setUTCDate(end.getUTCDate()-1);
    const start=new Date(end);start.setUTCDate(start.getUTCDate()-7);
    const schedule=await request(`${MLB}/api/v1/schedule?sportId=1&teamId=${id}&startDate=${start.toISOString().slice(0,10)}&endDate=${end.toISOString().slice(0,10)}`);
    const prior=(schedule.dates||[]).flatMap(d=>d.games||[]).filter(g=>g.status?.abstractGameState==='Final').sort((a,b)=>Date.parse(b.gameDate)-Date.parse(a.gameDate))[0];
    if (!prior) return null;
    const feed=await request(`${MLB}/api/v1.1/game/${prior.gamePk}/feed/live`);
    return {provider:'PREVIOUS_GAME',source:`MLB previous game ${prior.gamePk}`,timestamp:prior.gameDate,retrievedAt:stamp(),players:feedOrder(feed,prior.teams.away.team.id===id?'away':'home'),confirmed:false};
  })().catch(()=>null));
  return previousCache.get(key);
}
function snapshotSide(snapshot, gamePk,side) { return snapshot?.games?.[String(gamePk)]?.[side] || {}; }
function validateReports(rows, kind) {
  return (rows || []).filter(r=>['TEAM','BEAT','REPORTING'].includes(r.provider) && r.source && r.timestamp && r.verified===true && r.explicit===true && (kind!=='lineup' || r.confirmed===true) && (kind!=='starter' || !['TEAM','BEAT'].includes(r.provider) || r.confirmed===true));
}
export async function resolveGameInputs(game, feed, sources, previous = null) {
  const result={gamePk:game.gamePk,retrievedAt:stamp(),providers:[sources.lineups,sources.starters,sources.rr,sources.reports,sources.news].map(({value,...x})=>x)};
  for (const side of ['away','home']) {
    const rr=snapshotSide(sources.rr.value,game.gamePk,side), reports=snapshotSide(sources.reports.value,game.gamePk,side);
    const rw=matchedRW(sources.lineups.value,game,side), sp=matchedRW(sources.starters.value,game,side);
    const prior=await previousOrder(game,side,sources.date), order=feedOrder(feed,side), now=stamp();
    const reportNews=validateReports(reports.news,'news');
    const team=game.teams[side].team;
    const players=Object.values(feed.gameData?.players || {}).map(p=>norm(p.fullName));
    const news=[...reportNews,...(sources.news.value || []).filter(n=>norm(n.reason).includes(norm(team.teamName || team.name)) || players.some(p=>p.length>7 && norm(n.reason).includes(p)))];
    const bulk=sp?.primaryPitcher || rw?.primaryPitcher;
    if(bulk) news.push({source:'RotoWire public lineups',timestamp:now,action:'PRIMARY_BULK_PITCHER',reason:`${bulk} is labeled PRIM, not the starting pitcher`,recommendedAction:'SRM_REVIEW_RECOMMENDED'});
    if(bulk && !sp?.starter) news.push({source:'RotoWire public lineups',timestamp:now,action:'STARTER_UNRESOLVED',reason:'Public page identifies a bulk pitcher without identifying the actual opener',recommendedAction:'STARTER_UPDATE_REQUIRED'});
    if (sp?.opener) news.push({source:'RotoWire Projected Starters',timestamp:now,action:'OPENER',reason:`Reported opener: ${sp.opener}`,recommendedAction:'SRM_REVIEW_RECOMMENDED'});
    const opponent=feed.gameData?.probablePitchers?.[side==='away'?'home':'away'];
    const opposingHand=feed.gameData?.players?.[`ID${opponent?.id}`]?.pitchHand?.code;
    const rrLineup=rr.lineup && (!rr.lineup.vsHand || rr.lineup.vsHand===opposingHand) ? {...rr.lineup,provider:'ROSTERRESOURCE',confirmed:false} : null;
    const lineup=selectLineup([...validateReports(reports.lineups,'lineup'),...(rw?.lineup?[rw.lineup]:[]),...(rrLineup?[rrLineup]:[]),...(prior?[prior]:[]),
      ...(order.length===9?[{provider:'MLB',source:'MLB Stats API confirmed batting order',timestamp:now,retrievedAt:now,confirmed:true,players:order}]:[])],reportNews);
    const mlb=feed.gameData?.probablePitchers?.[side];
    const starter=selectStarter([...validateReports(reports.starters,'starter'),...(sp?.starter?[sp.starter]:[]),...(rr.starter?[{...rr.starter,provider:'ROSTERRESOURCE',confirmed:false}]:[]),
      ...(mlb?.id?[{provider:'MLB',source:'MLB Stats API probable pitcher',timestamp:now,retrievedAt:now,id:mlb.id,name:mlb.fullName,confirmed:false}]:[])]);
    result[side]={lineup,starter,news,srmReviews:srmReview(news,starter.name)};
  }
  result.gate=projectionGate({...result,previous});
  assertBaseballOnly(result);
  return result;
}
// Resolve exact MLB identities only. Provider IDs are NEVER treated as MLB IDs.
export async function applyResolvedInputs(feed, audit) {
  const output=structuredClone(feed);
  for (const side of ['away','home']) {
    const box=output.liveData?.boxscore?.teams?.[side];
    if (!box) throw new Error('MISSING_TEAM_BOX');
    const rosterId=output.gameData?.teams?.[side]?.id;
    const local=Object.values(output.gameData?.players || {});
    const roster=rosterId ? await safeSource('MLB_ROSTER',()=>request(`${MLB}/api/v1/teams/${rosterId}/roster?rosterType=40Man&hydrate=person`)) : {value:null};
    const people=[...local,...(roster.value?.roster || []).map(r=>r.person)];
    const resolve=name=>{
      const found=new Map(people.filter(p=>norm(p.fullName)===norm(name)).map(p=>[p.id,p]));
      if (found.size!==1) throw new Error(`UNRESOLVED_MLB_ID:${name}`);
      return [...found.values()][0];
    };
    for (const p of Object.values(box.players || {})) p.battingOrder='';
    for (const [index,name] of audit[side].lineup.players.entries()) {
      const p=resolve(name), key=`ID${p.id}`;
      output.gameData.players[key] ||= p;
      box.players[key] ||= {person:{id:p.id,fullName:p.fullName}};
      box.players[key].battingOrder=String((index+1)*100);
    }
    const starter=audit[side].starter;
    output.gameData.probablePitchers ||= {};
    if (starter.name) { const p=resolve(starter.name);output.gameData.probablePitchers[side]={id:p.id,fullName:p.fullName}; }
    else delete output.gameData.probablePitchers[side];
  }
  return output;
}
