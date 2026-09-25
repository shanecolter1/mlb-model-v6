import fs from 'node:fs';
import path from 'node:path';

const DATE = String(process.env.I2_DATE || new Date().toISOString().slice(0,10));
const SEASON = Number(DATE.slice(0,4));
const END = new Date(`${DATE}T12:00:00Z`);
END.setUTCDate(END.getUTCDate()-1);
const END_DATE = END.toISOString().slice(0,10);
const OUTPUT = String(process.env.I2_EMPIRICAL_OUTPUT || `data/runtime/i2/${DATE}_empirical_snapshot.json`);
const CONCURRENCY = Number(process.env.I2_EMPIRICAL_CONCURRENCY || 8);

async function getJson(url){
  const r=await fetch(url,{headers:{accept:'application/json','user-agent':'MLB-I2-Empirical-Snapshot/2.0'}});
  if(!r.ok) throw new Error(`${r.status} ${r.statusText}: ${url}`);
  return r.json();
}
async function mapLimit(items, limit, fn){
  const out=new Array(items.length); let next=0;
  async function worker(){
    while(true){
      const i=next++;
      if(i>=items.length) return;
      out[i]=await fn(items[i],i);
    }
  }
  await Promise.all(Array.from({length:Math.max(1,Math.min(limit,items.length||1))},worker));
  return out;
}
function bucket(r){ return r>=4?'4+':String(Math.max(0,Number(r)||0)); }
const KEYS=['0','1','2','3','4+'];
function dist(rows){
  const n=rows.length;
  const c=Object.fromEntries(KEYS.map(k=>[k,0]));
  for(const row of rows)c[bucket(row.i2Runs)]++;
  return {n,exact:Object.fromEntries(KEYS.map(k=>[k,n?c[k]/n:0]))};
}
function weighted(pitcherRows, offenseRows){
  const p=dist(pitcherRows), o=dist(offenseRows);
  if(!p.n || !o.n) return null;
  const exact=Object.fromEntries(KEYS.map(k=>[k,.65*p.exact[k]+.35*o.exact[k]]));
  return withCum({exact,pitcherN:p.n,offenseN:o.n});
}
function withCum(x){
  const e=x.exact;
  return {...x,cumulative:{
    '1+':1-e['0'],
    '2+':e['2']+e['3']+e['4+'],
    '3+':e['3']+e['4+'],
    '4+':e['4+'],
  }};
}
function convolve(a,b){
  if(!a||!b)return null;
  const e=Object.fromEntries(KEYS.map(k=>[k,0]));
  const vals={'0':0,'1':1,'2':2,'3':3,'4+':4};
  for(const ka of KEYS)for(const kb of KEYS){
    const r=Math.min(4,vals[ka]+vals[kb]);
    e[r>=4?'4+':String(r)] += a.exact[ka]*b.exact[kb];
  }
  return withCum({exact:e});
}
function pct(v){ return v==null?null:Number((100*v).toFixed(2)); }
function pctDist(d){
  if(!d)return null;
  return {
    ...d,
    exact:Object.fromEntries(Object.entries(d.exact).map(([k,v])=>[k,pct(v)])),
    cumulative:Object.fromEntries(Object.entries(d.cumulative).map(([k,v])=>[k,pct(v)])),
  };
}
function fair(p){
  if(!(p>0&&p<1))return null;
  return Math.round(p>=.5?-100*p/(1-p):100*(1-p)/p);
}

const feedCache=new Map();
async function feed(pk){
  if(!feedCache.has(pk))feedCache.set(pk,getJson(`https://statsapi.mlb.com/api/v1.1/game/${pk}/feed/live`));
  return feedCache.get(pk);
}

const pitcherCache=new Map();
async function pitcherData(p){
  if(!p?.id)return null;
  if(pitcherCache.has(p.id))return pitcherCache.get(p.id);
  const promise=(async()=>{
    const gl=await getJson(`https://statsapi.mlb.com/api/v1/people/${p.id}/stats?stats=gameLog&group=pitching&season=${SEASON}`);
    const starts=(gl.stats?.[0]?.splits||[]).filter(s=>Number(s.stat?.gamesStarted||0)>0&&s.game?.gamePk);
    const rows=(await mapLimit(starts,CONCURRENCY,async s=>{
      const f=await feed(s.game.gamePk);
      if(f.gameData?.game?.type!=='R')return null;
      const d=String(f.gameData?.datetime?.officialDate||s.date||'');
      if(!d || d>=DATE)return null;
      let runs=0,faced=false;
      for(const play of f.liveData?.plays?.allPlays||[]){
        if(play.about?.inning!==2||play.matchup?.pitcher?.id!==p.id)continue;
        faced=true;
        for(const rr of play.runners||[])if(rr.movement?.end==='score')runs++;
      }
      return faced?{gamePk:s.game.gamePk,date:d,i2Runs:runs}:null;
    })).filter(Boolean).sort((a,b)=>a.date.localeCompare(b.date));
    return {id:p.id,name:p.fullName||p.name,rows,season:dist(rows),last15:dist(rows.slice(-15)),last10:dist(rows.slice(-10)),last5:dist(rows.slice(-5))};
  })();
  pitcherCache.set(p.id,promise); return promise;
}

const offenseCache=new Map();
async function offenseData(team){
  if(offenseCache.has(team.id))return offenseCache.get(team.id);
  const promise=(async()=>{
    const s=await getJson(`https://statsapi.mlb.com/api/v1/schedule?sportId=1&teamId=${team.id}&startDate=${SEASON}-03-25&endDate=${END_DATE}&hydrate=linescore`);
    const games=[];
    for(const d of s.dates||[])for(const g of d.games||[]){
      if(!['Final','Game Over','Completed Early'].includes(g.status?.detailedState))continue;
      if(g.gameType!=='R')continue;
      games.push(g);
    }
    games.sort((a,b)=>Date.parse(a.gameDate)-Date.parse(b.gameDate));
    const rows=(await mapLimit(games,CONCURRENCY,async g=>{
      const isAway=g.teams?.away?.team?.id===team.id;
      let rr=g.linescore?.innings?.find(x=>x.num===2)?.[isAway?'away':'home']?.runs;
      if(rr==null){
        const f=await feed(g.gamePk);
        rr=(f.liveData?.linescore?.innings||[]).find(x=>x.num===2)?.[isAway?'away':'home']?.runs;
      }
      return rr==null?null:{gamePk:g.gamePk,date:g.officialDate||g.gameDate?.slice(0,10),i2Runs:Number(rr)};
    })).filter(Boolean);
    return {id:team.id,name:team.name,rows,season:dist(rows),last25:dist(rows.slice(-25)),last10:dist(rows.slice(-10))};
  })();
  offenseCache.set(team.id,promise); return promise;
}

const schedule=await getJson(`https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=${DATE}&hydrate=probablePitcher,team`);
const scheduled=(schedule.dates||[]).flatMap(d=>d.games||[]).filter(g=>g.gameType==='R');
const metas=await mapLimit(scheduled,CONCURRENCY,async g=>{
  const f=await feed(g.gamePk);
  return {
    gamePk:g.gamePk,
    gameDate:g.gameDate,
    away:f.gameData?.teams?.away||g.teams?.away?.team,
    home:f.gameData?.teams?.home||g.teams?.home?.team,
    awayStarter:f.gameData?.probablePitchers?.away||g.teams?.away?.probablePitcher||null,
    homeStarter:f.gameData?.probablePitchers?.home||g.teams?.home?.probablePitcher||null,
  };
});

const out={
  date:DATE,
  generatedAt:new Date().toISOString(),
  method:{
    locked05:'Half-inning scoreless = 65% starting-pitcher season I2 scoreless rate + 35% opponent offense last-25 I2 scoreless rate. Full I2 Under 0.5 = top scoreless x bottom scoreless.',
    distributionExtension:'For alternate run totals only, the same 65/35 weights are applied bucket-by-bucket to observed I2 run distributions (0,1,2,3,4+); full-inning distribution is the convolution of top and bottom distributions. This extension does not alter the locked 0.5 formula.',
    recentWindows:'Pitcher last15/10/5 and offense last10 are diagnostics only and are not model inputs.'
  },
  games:[]
};

await mapLimit(metas,Math.min(CONCURRENCY,4),async meta=>{
  const away={id:meta.away?.id,name:meta.away?.name};
  const home={id:meta.home?.id,name:meta.home?.name};
  if(!meta.awayStarter?.id||!meta.homeStarter?.id){
    out.games.push({gamePk:meta.gamePk,gameDate:meta.gameDate,matchup:`${away.name} @ ${home.name}`,error:'MISSING_PROBABLE_STARTER'});
    return;
  }
  const [aPit,hPit,aOff,hOff]=await Promise.all([
    pitcherData(meta.awayStarter),pitcherData(meta.homeStarter),offenseData(away),offenseData(home)
  ]);
  const top=weighted(hPit.rows,aOff.rows.slice(-25));
  const bottom=weighted(aPit.rows,hOff.rows.slice(-25));
  const full=convolve(top,bottom);
  if(!top||!bottom||!full){
    out.games.push({gamePk:meta.gamePk,gameDate:meta.gameDate,matchup:`${away.name} @ ${home.name}`,error:'INSUFFICIENT_EMPIRICAL_SAMPLE'});
    return;
  }
  const under=top.exact['0']*bottom.exact['0'];
  const over=1-under;
  out.games.push({
    gamePk:meta.gamePk,gameDate:meta.gameDate,matchup:`${away.name} @ ${home.name}`,
    awayTeam:away.name,homeTeam:home.name,
    awayStarter:aPit.name,homeStarter:hPit.name,
    top2:pctDist(top),bottom2:pctDist(bottom),fullI2:pctDist(full),
    top2ScorelessPct:pct(top.exact['0']),bottom2ScorelessPct:pct(bottom.exact['0']),
    top2ScorePct:pct(1-top.exact['0']),bottom2ScorePct:pct(1-bottom.exact['0']),
    empiricalUnder05Pct:pct(under),empiricalOver05Pct:pct(over),
    empiricalFairUnder:fair(under),empiricalFairOver:fair(over),
    diagnostics:{
      awayStarter:{season:pctDist(withCum(aPit.season)),last15:pctDist(withCum(aPit.last15)),last10:pctDist(withCum(aPit.last10)),last5:pctDist(withCum(aPit.last5))},
      homeStarter:{season:pctDist(withCum(hPit.season)),last15:pctDist(withCum(hPit.last15)),last10:pctDist(withCum(hPit.last10)),last5:pctDist(withCum(hPit.last5))},
      awayOffense:{last25:pctDist(withCum(dist(aOff.rows.slice(-25)))),last10:pctDist(withCum(dist(aOff.rows.slice(-10))))},
      homeOffense:{last25:pctDist(withCum(dist(hOff.rows.slice(-25)))),last10:pctDist(withCum(dist(hOff.rows.slice(-10))))},
    }
  });
});
out.games.sort((a,b)=>String(a.gameDate).localeCompare(String(b.gameDate))||Number(a.gamePk)-Number(b.gamePk));
fs.mkdirSync(path.dirname(OUTPUT),{recursive:true});
fs.writeFileSync(OUTPUT,JSON.stringify(out,null,2)+'\n');
console.log(JSON.stringify({output:OUTPUT,games:out.games.length,modeled:out.games.filter(x=>!x.error).length,errors:out.games.filter(x=>x.error).map(x=>({gamePk:x.gamePk,matchup:x.matchup,error:x.error}))},null,2));
