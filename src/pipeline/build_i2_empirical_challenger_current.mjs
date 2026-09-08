import fs from 'node:fs';

const date=process.env.I2_DATE||new Date().toISOString().slice(0,10);
async function j(u){const r=await fetch(u);if(!r.ok)throw new Error(`${r.status} ${u}`);return r.json();}
const audit=JSON.parse(fs.readFileSync(`data/runtime/i2/${date}_official_lineup_audit.json`,'utf8'));
const offense=JSON.parse(fs.readFileSync('data/derived/i2/2026_team_second_inning_runs.json','utf8'));
function off(team){const r=offense.rows.find(x=>x.team===team);return r?{seasonUnder:1-r.i2ScorePct,last25Under:r.last25?.i2UnderPct??null,last25Runs:r.last25?.i2Runs??null,last25ScoreGames:r.last25?.i2ScoreGames??null}:null;}
async function pitcher(p){
  if(!p?.id)return null;
  const gl=await j(`https://statsapi.mlb.com/api/v1/people/${p.id}/stats?stats=gameLog&group=pitching&season=2026`);
  const splits=(gl.stats?.[0]?.splits||[]).filter(s=>Number(s.stat?.gamesStarted||0)>0&&s.game?.gamePk);
  const rows=[];
  for(const s of splits){
    const g=await j(`https://statsapi.mlb.com/api/v1.1/game/${s.game.gamePk}/feed/live`);
    if(g.gameData?.game?.type&&g.gameData.game.type!=='R')continue;
    let runs=0,faced=false;
    for(const play of g.liveData?.plays?.allPlays||[]){
      if(play.about?.inning!==2||play.matchup?.pitcher?.id!==p.id)continue;
      faced=true;
      for(const rr of play.runners||[])if(rr.movement?.end==='score')runs++;
    }
    if(faced)rows.push({gamePk:s.game.gamePk,date:s.date,runs,scoreless:runs===0});
  }
  rows.sort((a,b)=>a.date.localeCompare(b.date));
  const n=rows.length,scoreless=rows.filter(x=>x.scoreless).length;
  return {id:p.id,name:p.name,n,scoreless,scorelessPct:n?scoreless/n:null,i2Runs:rows.reduce((a,b)=>a+b.runs,0),last5ScorelessPct:n?rows.slice(-5).filter(x=>x.scoreless).length/Math.min(5,n):null,last10ScorelessPct:n?rows.slice(-10).filter(x=>x.scoreless).length/Math.min(10,n):null};
}
const results=[];
for(const g of audit.games){
  if(!g.lineupsConfirmed)continue;
  const aP=await pitcher(g.awayStarter),hP=await pitcher(g.homeStarter);
  const [awayTeam,homeTeam]=g.matchup.split(' @ '),aOff=off(awayTeam),hOff=off(homeTeam);
  const top=hP&&aOff&&hP.scorelessPct!=null&&aOff.last25Under!=null?.65*hP.scorelessPct+.35*aOff.last25Under:null;
  const bot=aP&&hOff&&aP.scorelessPct!=null&&hOff.last25Under!=null?.65*aP.scorelessPct+.35*hOff.last25Under:null;
  results.push({gamePk:g.gamePk,matchup:g.matchup,awayStarter:aP,homeStarter:hP,awayOffense:aOff,homeOffense:hOff,top2Scoreless:top,bottom2Scoreless:bot,fullUnder:top!=null&&bot!=null?top*bot:null});
}
const out={date,generatedAt:new Date().toISOString(),method:'65% exact starter 2026 regular-season I2 scoreless + 35% opponent offense last-25 I2 scoreless; research challenger only',games:results};
fs.writeFileSync(`data/runtime/i2/${date}_official_empirical_challenger.json`,JSON.stringify(out,null,2)+'\n');
console.log(JSON.stringify(results.map(x=>({matchup:x.matchup,awayStarter:`${x.awayStarter?.scoreless}/${x.awayStarter?.n}`,homeStarter:`${x.homeStarter?.scoreless}/${x.homeStarter?.n}`,top2:x.top2Scoreless,bottom2:x.bottom2Scoreless,full:x.fullUnder})),null,2));
