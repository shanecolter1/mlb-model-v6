import fs from 'node:fs';
import path from 'node:path';

const DATE = process.env.I2_DATE || new Date().toISOString().slice(0,10);
const OUTPUT = process.env.I2_LINEUP_OVERRIDES || `data/runtime/i2/${DATE}_lineup_overrides.json`;
const LOOKBACK_DAYS = Number(process.env.I2_PRELIM_LOOKBACK_DAYS || 7);

function isoDate(d){ return d.toISOString().slice(0,10); }
function addDays(s,n){ const d=new Date(`${s}T12:00:00Z`); d.setUTCDate(d.getUTCDate()+n); return isoDate(d); }
async function getJson(url){
  const r=await fetch(url,{headers:{accept:'application/json','user-agent':'MLB-I2-Preliminary-Lineups/1.0'}});
  if(!r.ok) throw new Error(`${r.status} ${r.statusText}: ${url}`);
  return r.json();
}
function lineupFromFeed(feed, side){
  const players=Object.values(feed?.liveData?.boxscore?.teams?.[side]?.players||{});
  const starters=players
    .filter(p=>Number(p?.battingOrder)>0 && Number(p.battingOrder)%100===0)
    .sort((a,b)=>Number(a.battingOrder)-Number(b.battingOrder))
    .slice(0,9)
    .map(p=>p?.person?.fullName)
    .filter(Boolean);
  return starters.length===9?starters:null;
}

const today=await getJson(`https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=${DATE}`);
const todayGames=(today?.dates||[]).flatMap(d=>d.games||[]).filter(g=>Number(g?.gameType==='R'?1:1));
if(!todayGames.length) throw new Error(`No MLB games found for ${DATE}`);

const start=addDays(DATE,-LOOKBACK_DAYS), end=addDays(DATE,-1);
const hist=await getJson(`https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate=${start}&endDate=${end}`);
const histGames=(hist?.dates||[]).flatMap(d=>d.games||[])
  .filter(g=>['Final','Game Over','Completed Early'].includes(String(g?.status?.detailedState||'')))
  .sort((a,b)=>Date.parse(b.gameDate)-Date.parse(a.gameDate));

const latestByTeam=new Map();
for(const g of histGames){
  for(const side of ['away','home']){
    const id=Number(g?.teams?.[side]?.team?.id);
    if(id && !latestByTeam.has(id)) latestByTeam.set(id,{gamePk:g.gamePk,side,gameDate:g.gameDate,opponent:g?.teams?.[side==='away'?'home':'away']?.team?.name||null});
  }
}

const feedCache=new Map();
async function priorLineup(teamId){
  const ref=latestByTeam.get(Number(teamId));
  if(!ref) return {lineup:null,ref:null,error:'NO_PRIOR_COMPLETED_GAME'};
  try{
    if(!feedCache.has(ref.gamePk)) feedCache.set(ref.gamePk,await getJson(`https://statsapi.mlb.com/api/v1.1/game/${ref.gamePk}/feed/live`));
    const lineup=lineupFromFeed(feedCache.get(ref.gamePk),ref.side);
    return {lineup,ref,error:lineup?null:'PRIOR_GAME_LINEUP_INCOMPLETE'};
  }catch(e){ return {lineup:null,ref,error:String(e)}; }
}

const games={};
const audit=[];
for(const g of todayGames){
  const awayId=g?.teams?.away?.team?.id, homeId=g?.teams?.home?.team?.id;
  const away=await priorLineup(awayId), home=await priorLineup(homeId);
  const matchup=`${g?.teams?.away?.team?.name} @ ${g?.teams?.home?.team?.name}`;
  if(away.lineup?.length===9 && home.lineup?.length===9){
    games[String(g.gamePk)]={matchup,away:away.lineup,home:home.lineup};
  }
  audit.push({
    gamePk:g.gamePk, matchup,
    awayTeamId:awayId, homeTeamId:homeId,
    awayPriorGamePk:away.ref?.gamePk||null, awayPriorGameDate:away.ref?.gameDate||null, awayError:away.error,
    homePriorGamePk:home.ref?.gamePk||null, homePriorGameDate:home.ref?.gameDate||null, homeError:home.error,
    usable:Boolean(away.lineup?.length===9 && home.lineup?.length===9)
  });
}

const out={
  date:DATE,
  generatedAt:new Date().toISOString(),
  governance:{
    sourceStatus:'PRELIMINARY_BASEBALL_ONLY',
    methodology:'Use each team\'s most recent completed MLB game starting batting order as the provisional batting order when today\'s MLB feed does not yet contain a complete confirmed order. MLB feed-confirmed orders always override this file.',
    source:'MLB Stats API only',
    marketDataUsed:false,
    note:'Preliminary lineups are explicitly provisional and must be replaced by confirmed MLB batting orders when available.'
  },
  lookback:{startDate:start,endDate:end,days:LOOKBACK_DAYS},
  games,
  audit
};
fs.mkdirSync(path.dirname(OUTPUT),{recursive:true});
fs.writeFileSync(OUTPUT,JSON.stringify(out,null,2));
console.log(JSON.stringify({output:OUTPUT,todayGames:todayGames.length,usableOverrides:Object.keys(games).length,unusable:audit.filter(x=>!x.usable)},null,2));
