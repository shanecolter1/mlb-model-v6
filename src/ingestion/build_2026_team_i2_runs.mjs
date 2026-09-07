import fs from 'node:fs';
const SEASON=2026;
const END=process.env.END_DATE||new Date().toISOString().slice(0,10);
async function j(u){const r=await fetch(u,{headers:{accept:'application/json','user-agent':'MLB-I2-TeamRates/1.2'}});if(!r.ok)throw new Error(`${r.status} ${u}`);return r.json()}
const sched=await j(`https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate=${SEASON}-03-01&endDate=${END}&gameTypes=R`);
const games=(sched.dates||[]).flatMap(d=>d.games||[]).filter(g=>String(g.status?.abstractGameState||'').toLowerCase()==='final');
const agg=new Map();
const history=new Map();
function get(team){if(!agg.has(team.id))agg.set(team.id,{teamId:team.id,team:team.name,games:0,i2Runs:0,i2ScoreGames:0});if(!history.has(team.id))history.set(team.id,[]);return agg.get(team.id)}
function record(teamId,g,runs){history.get(teamId).push({gamePk:g.gamePk,officialDate:g.officialDate||null,runs:Number(runs||0),scored:Number(runs||0)>0});}
async function processGame(g){
 const feed=await j(`https://statsapi.mlb.com/api/v1.1/game/${g.gamePk}/feed/live`);
 const inn=(feed.liveData?.linescore?.innings||[]).find(x=>Number(x.num)===2);
 if(!inn)return;
 const a=get(feed.gameData.teams.away),h=get(feed.gameData.teams.home);
 const ar=Number(inn.away?.runs||0),hr=Number(inn.home?.runs||0);
 a.games++;h.games++;a.i2Runs+=ar;h.i2Runs+=hr;if(ar>0)a.i2ScoreGames++;if(hr>0)h.i2ScoreGames++;
 record(a.teamId,g,ar);record(h.teamId,g,hr);
}
const CONCURRENCY=25;
for(let i=0;i<games.length;i+=CONCURRENCY){
 await Promise.all(games.slice(i,i+CONCURRENCY).map(processGame));
 console.log(Math.min(i+CONCURRENCY,games.length),'/',games.length);
}
function trailing25(teamId){
 const h=[...(history.get(teamId)||[])].sort((a,b)=>String(a.officialDate).localeCompare(String(b.officialDate))||Number(a.gamePk)-Number(b.gamePk)).slice(-25);
 const scoreGames=h.filter(x=>x.scored).length;
 const runs=h.reduce((s,x)=>s+x.runs,0);
 return {
  games:h.length,
  i2Runs:runs,
  i2ScoreGames:scoreGames,
  i2ScorePct:h.length?scoreGames/h.length:null,
  i2UnderPct:h.length?(h.length-scoreGames)/h.length:null,
  fromDate:h[0]?.officialDate||null,
  throughDate:h.at(-1)?.officialDate||null,
  gamesDetail:h
 };
}
const rows=[...agg.values()].map(x=>({...x,i2RunsPerGame:x.games?x.i2Runs/x.games:null,i2ScorePct:x.games?x.i2ScoreGames/x.games:null,last25:trailing25(x.teamId)})).sort((a,b)=>b.i2RunsPerGame-a.i2RunsPerGame);
fs.mkdirSync('data/derived/i2',{recursive:true});
const csv=['rank,team,games,i2_runs,i2_runs_per_game,i2_score_games,i2_score_pct'];
rows.forEach((x,i)=>csv.push([i+1,`"${x.team.replaceAll('"','""')}"`,x.games,x.i2Runs,x.i2RunsPerGame.toFixed(4),x.i2ScoreGames,(100*x.i2ScorePct).toFixed(2)].join(',')));
fs.writeFileSync('data/derived/i2/2026_team_second_inning_runs.csv',csv.join('\n')+'\n');
fs.writeFileSync('data/derived/i2/2026_team_second_inning_runs.json',JSON.stringify({season:SEASON,endDate:END,finalGames:games.length,rows},null,2));
console.log(csv.join('\n'));
for(const x of rows){if(['St. Louis Cardinals','San Francisco Giants'].includes(x.team))console.log(JSON.stringify({team:x.team,last25:x.last25},null,2));}
