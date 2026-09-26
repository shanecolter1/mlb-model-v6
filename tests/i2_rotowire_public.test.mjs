import test from 'node:test';
import assert from 'node:assert/strict';
import {parseRotowirePublic,fetchRotowirePublic} from '../src/inputs/rotowire_public.mjs';
import {collectSources} from '../src/inputs/i2_baseball_sources.mjs';
const date='2026-09-26',stamp='2026-09-26T12:00:00Z';
function card(time='1:10 PM ET',status='expected',primary=false) {
 return `<div class="lineup is-mlb"><div class="lineup__meta"><div class="lineup__time">${time}</div></div><div class="lineup__teams"><div class="lineup__abbr">NYY</div><div class="lineup__abbr">BOS</div></div>`+['visit','home'].map(side=>`<ul class="lineup__list is-${side}"><li class="lineup__player-highlight"><div class="lineup__player-highlight-name"><a href="/baseball/player/pitcher-1">${side} Starter</a><span class="lineup__throws">R</span></div><div class="lineup__player-highlight-stats">ERA ${primary?'<div class="tag">PRIM</div>':''}</div></li><li class="lineup__status is-${status}"><div class="dot"></div>${status==='confirmed'?'Confirmed':'Expected'} Lineup</li>`+Array.from({length:9},(_,i)=>`<li class="lineup__player"><div class="lineup__pos">DH</div><a href="/baseball/player/hitter-${i}" title="${side} Hitter ${i}">H. ${i}</a><span class="lineup__bats">L</span></li>`).join('')+'</ul>').join('')+'<div class="lineup__odds-item">LINE NYY -125 O/U 8.5</div></div>';
}
const page=(content=card())=>`<main data-sportfull="baseball" data-gamedate="${date}">${content}</main>`;
test('public expected lineup full names, order, hand, starter, ET time and retrieval timestamp',()=>{
 const g=parseRotowirePublic(page(),date,stamp)[0];assert.equal(g.gameDate,'2026-09-26T17:10:00.000Z');assert.equal(g.teams[0].lineup.players.length,9);assert.equal(g.teams[0].lineup.players[0],'visit Hitter 0');assert.equal(g.teams[0].lineup.confirmed,false);assert.equal(g.teams[0].lineup.retrievedAt,stamp);assert.equal(g.teams[0].lineup.timestamp,null);assert.equal(g.teams[0].starter.name,'visit Starter');
});
test('explicit confirmed marker accepted; unknown status never promoted',()=>{
 assert.equal(parseRotowirePublic(page(card('1:10 PM ET','confirmed')),date)[0].teams[0].lineup.confirmed,true);
 assert.equal(parseRotowirePublic(page(card('1:10 PM ET','unknown')),date)[0].teams[0].lineup,null);
});
test('PRIM is a bulk pitcher and is never selected as the starter',()=>{
 const t=parseRotowirePublic(page(card('1:10 PM ET','expected',true)),date)[0].teams[0];assert.equal(t.starter,null);assert.equal(t.primaryPitcher,'visit Starter');
});
test('market blocks and scripts cannot influence or enter normalized input objects',()=>{
 const a=parseRotowirePublic(page(),date,stamp),b=parseRotowirePublic(page().replace('LINE NYY -125 O/U 8.5','moneyline -9000 odds 99999 sportsbook FanDuel')+'<script>var americanOdds=900</script>',date,stamp);
 assert.deepEqual(a,b);assert.doesNotMatch(JSON.stringify(a),/odds|moneyline|sportsbook|FanDuel|O\/U/i);
});
test('wrong dates, login pages and unsupported markup fail safely',()=>{
 assert.throws(()=>parseRotowirePublic(page(),'2026-09-27'),/DATE_MISMATCH/);assert.throws(()=>parseRotowirePublic('<h1>Sign in</h1>',date));assert.throws(()=>parseRotowirePublic(page('changed markup'),date),/SCHEMA/);
});
test('doubleheader cards preserve distinct times; winter ET uses standard time',()=>{
 const games=parseRotowirePublic(page(card()+card('7:10 PM ET')),date);assert.equal(games.length,2);assert.notEqual(games[0].gameDate,games[1].gameDate);
 assert.equal(parseRotowirePublic(page().replaceAll(date,'2026-01-26'),'2026-01-26')[0].gameDate,'2026-01-26T18:10:00.000Z');
});
test('missing player and duplicate player invalidate lineup without inventing identities',()=>{
 assert.equal(parseRotowirePublic(page().replace('title="visit Hitter 0"','title="visit Hitter 1"'),date)[0].teams[0].lineup,null);
});
test('no API key required; one page request serves lineups and starters; key remains unused on success',async()=>{
 const original=globalThis.fetch;let pages=0,apis=0;
 globalThis.fetch=async u=>{if(String(u).includes('api.rotowire.com')){apis++;throw new Error('API must not be used');}if(String(u).includes('rotowire.com')){pages++;return new Response(page());}return new Response('<rss/>');};
 try {for(const env of [{},{ROTOWIRE_API_KEY:'unused'}]){const x=await collectSources(date,env);assert.equal(x.lineups.status,'AVAILABLE');assert.equal(x.starters.status,'AVAILABLE');assert.equal(x.lineups.value[0].teams[0].lineup.transport,'PUBLIC_HTML');}assert.equal(pages,2);assert.equal(apis,0);}finally{globalThis.fetch=original;}
});
test('public access failure never falls back to a paid RotoWire API',async()=>{
 const original=globalThis.fetch;let apiCalls=0;
 globalThis.fetch=async u=>{if(String(u).includes('api.rotowire.com')){apiCalls++;throw new Error('paid RotoWire API must not be called');}return new Response('unavailable',{status:403});};
 try{
  const x=await collectSources(date,{ROTOWIRE_API_KEY:'ignored'});
  assert.equal(x.lineups.status,'ROTOWIRE_UNAVAILABLE');
  assert.equal(x.starters.status,'ROTOWIRE_STARTERS_UNAVAILABLE');
  assert.equal(x.lineups.publicPage.status,'ROTOWIRE_PUBLIC_UNAVAILABLE');
  assert.equal(apiCalls,0);
 }finally{globalThis.fetch=original;}
});
test('GitHub runner public access without credentials',{skip:process.env.I2_TEST_LIVE_ROTOWIRE!=='1'},async()=>{
 const today=new Intl.DateTimeFormat('en-CA',{timeZone:'America/New_York',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date());
 const x=await fetchRotowirePublic(today);assert.ok(x.some(g=>g.teams.some(t=>t.lineup)));
 console.log(JSON.stringify({transport:'PUBLIC_HTML',apiKeyUsed:false,date:today,games:x.length,lineups:x.flatMap(g=>g.teams).filter(t=>t.lineup).length,starters:x.flatMap(g=>g.teams).filter(t=>t.starter).length}));
});
