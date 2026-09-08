import fs from 'node:fs';

const date=process.env.I2_DATE||new Date().toISOString().slice(0,10);
async function j(u){const r=await fetch(u);if(!r.ok)throw new Error(`${r.status} ${u}`);return r.json();}
const sched=await j(`https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=${date}`);
const games=sched.dates?.[0]?.games||[];
const overrides={date,generatedAt:new Date().toISOString(),governance:{sourceStatus:'MLB_CONFIRMED_LINEUPS_ONLY',methodology:'MLB Stats API live-feed battingOrder arrays. No previous-game or projected batting order is used. Games without two nine-man MLB batting orders remain unconfirmed.',source:'MLB Stats API schedule + live feed',marketDataUsed:false},games:{}};
const audit=[];
async function starterStats(id){
  if(!id)return null;
  try{const x=await j(`https://statsapi.mlb.com/api/v1/people/${id}/stats?stats=season&group=pitching&season=2026`);const s=x.stats?.[0]?.splits?.[0]?.stat||{};return {era:s.era??null,whip:s.whip??null,inningsPitched:s.inningsPitched??null,gamesStarted:s.gamesStarted??null,strikeOuts:s.strikeOuts??null,baseOnBalls:s.baseOnBalls??null,homeRuns:s.homeRuns??null};}catch{return null;}
}
for(const sg of games){
  const feed=await j(`https://statsapi.mlb.com/api/v1.1/game/${sg.gamePk}/feed/live`);
  const gd=feed.gameData||{},bs=feed.liveData?.boxscore?.teams||{};
  const playerName=id=>gd.players?.[`ID${id}`]?.fullName||null;
  const awayIds=Array.isArray(bs.away?.battingOrder)?bs.away.battingOrder:[];
  const homeIds=Array.isArray(bs.home?.battingOrder)?bs.home.battingOrder:[];
  const away=awayIds.map(playerName).filter(Boolean),home=homeIds.map(playerName).filter(Boolean);
  const ap=gd.probablePitchers?.away||null,hp=gd.probablePitchers?.home||null;
  const confirmed=away.length===9&&home.length===9;
  if(confirmed)overrides.games[String(sg.gamePk)]={matchup:`${sg.teams.away.team.name} @ ${sg.teams.home.team.name}`,lineupMethod:'MLB Stats API confirmed batting order',away,home};
  audit.push({gamePk:sg.gamePk,gameDate:sg.gameDate,matchup:`${sg.teams.away.team.name} @ ${sg.teams.home.team.name}`,abstractGameState:sg.status?.abstractGameState,detailedState:sg.status?.detailedState,lineupsConfirmed:confirmed,awayLineup:away,homeLineup:home,awayStarter:ap?{id:ap.id,name:ap.fullName,stats:await starterStats(ap.id)}:null,homeStarter:hp?{id:hp.id,name:hp.fullName,stats:await starterStats(hp.id)}:null});
}
fs.writeFileSync(`data/runtime/i2/${date}_official_lineups.json`,JSON.stringify(overrides,null,2)+'\n');
fs.writeFileSync(`data/runtime/i2/${date}_official_lineup_audit.json`,JSON.stringify({date,generatedAt:new Date().toISOString(),scheduledGames:games.length,confirmedGames:audit.filter(x=>x.lineupsConfirmed).length,games:audit},null,2)+'\n');
console.log(JSON.stringify({scheduled:games.length,confirmed:audit.filter(x=>x.lineupsConfirmed).length,games:audit.map(x=>({gamePk:x.gamePk,matchup:x.matchup,confirmed:x.lineupsConfirmed,awayStarter:x.awayStarter?.name,homeStarter:x.homeStarter?.name}))},null,2));
