// Offline production-boundary fixture: no network and no sportsbook information.
const date=process.env.I2_DATE;
const players={};
const teams={};
for (const [index,side] of ['away','home'].entries()) {
 const team={id:index+1,name:`${side} Team`,abbreviation:side==='away'?'AWY':'HME'};
 const box={team,players:{}};
 for(let i=1;i<=10;i++) {
  const id=(index+1)*100+i,fullName=`${side} Player ${i}`;
  players[`ID${id}`]={id,fullName,batSide:{code:i%2?'R':'L'}};
  box.players[`ID${id}`]={person:{id,fullName},battingOrder:i<=9?String(i*100):''};
 }
 teams[side]=box;
}
const game={gamePk:999999,gameDate:`${date}T23:59:59Z`,status:{abstractGameState:'Preview',detailedState:'Scheduled'},teams:{away:{team:teams.away.team},home:{team:teams.home.team}},venue:{name:'Fixture Park'}};
const feed={gameData:{teams:{away:teams.away.team,home:teams.home.team},players,probablePitchers:{away:{id:110,fullName:'away Player 10'},home:{id:210,fullName:'home Player 10'}}},liveData:{boxscore:{teams}}};
let feeds=0;
globalThis.fetch=async input=>{
 const u=new URL(String(input));
 if (/odds|market|sportsbook/.test(u.hostname)) throw new Error('MARKET_ACCESS_FORBIDDEN');
 if(u.hostname==='www.mlb.com')return new Response('<rss><channel></channel></rss>');
 if(u.pathname.includes('schedule'))return Response.json({dates:u.searchParams.has('startDate')?[]:[{games:[game]}]});
 if(u.pathname.includes('/feed/live')) {
  feeds++;const f=structuredClone(feed);
  if (process.env.I2_TEST_SOURCE_MODE==='provisional') for(const side of ['away','home']) for(const p of Object.values(f.liveData.boxscore.teams[side].players))p.battingOrder='';
  if (process.env.I2_TEST_SOURCE_MODE==='change' && feeds>1) f.gameData.probablePitchers.away={id:109,fullName:'away Player 9'};
  return Response.json(f);
 }
 if(u.pathname.includes('/roster'))return Response.json({roster:Object.values(players).map(person=>({person}))});
 if(u.pathname.includes('/stats'))return Response.json({stats:[{splits:[{stat:{plateAppearances:400,atBats:350,hits:100,doubles:20,triples:2,homeRuns:15,baseOnBalls:40,hitByPitch:5,strikeOuts:90,battersFaced:400,inningsPitched:'90.0',gamesStarted:20,era:'3.80',whip:'1.20'}}]}]});
 if(u.hostname==='www.rotowire.com') {
  if (process.env.I2_TEST_SOURCE_MODE !== 'provisional') return new Response('unavailable',{status:503});
  const list=(side,klass)=>`
    <ul class="lineup__list ${klass}">
      <li class="lineup__status is-expected">Expected Lineup</li>
      ${Array.from({length:9},(_,i)=>`<li class="lineup__player"><a href="/baseball/player/${side}-${i+1}" title="${side} Player ${i+1}">${side} Player ${i+1}</a><span class="lineup__bats">${i%2?'L':'R'}</span></li>`).join('')}
      <li class="lineup__player-highlight"><a href="/baseball/player/${side}-10" title="${side} Player 10">${side} Player 10</a></li>
    </ul>`;
  const html=`<main data-sportfull="baseball" data-gamedate="${date}">
    <div class="lineup is-mlb">
      <div class="lineup__time">7:59 PM ET</div>
      <div class="lineup__abbr">AWY</div><div class="lineup__abbr">HME</div>
      ${list('away','is-visit')}
      ${list('home','is-home')}
    </div>
  </main>`;
  return new Response(html,{status:200,headers:{'content-type':'text/html'}});
 }
 throw new Error(`UNEXPECTED_FIXTURE_REQUEST:${u.pathname}`);
};
