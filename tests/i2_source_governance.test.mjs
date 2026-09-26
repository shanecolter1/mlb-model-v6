import test from 'node:test';
import assert from 'node:assert/strict';
import { selectLineup,selectStarter,projectionGate,provisionalConfidence,lineupDelta,assertBaseballOnly,srmReview,sameMlbIdentityName } from '../src/inputs/i2_source_governance.mjs';
import { safeSource,normalizeRotowire,parseNews,collectSources } from '../src/inputs/i2_baseball_sources.mjs';
import { recommendationGate,enrichPriceDependentRecommendation } from '../src/model/i2_price_dependent_recommendations.mjs';
const timestamp=new Date().toISOString(), players=Array.from({length:9},(_,i)=>`Player ${i+1}`);
const lineup=(provider,extra={})=>({provider,source:provider,timestamp,retrievedAt:timestamp,players,confirmed:false,...extra});
const pitcher=(provider,name='Starter A',extra={})=>({provider,source:provider,timestamp,retrievedAt:timestamp,name,...extra});
const side=()=>({lineup:selectLineup([lineup('ROTOWIRE')]),starter:selectStarter([pitcher('ROTOWIRE')]),news:[]});
test('RotoWire takes priority over previous game; fallback only without valid RotoWire',()=>{
 assert.equal(selectLineup([lineup('PREVIOUS_GAME'),lineup('ROTOWIRE')]).status,'PROVISIONAL_ROTOWIRE');
 assert.equal(selectLineup([lineup('PREVIOUS_GAME'),lineup('ROTOWIRE',{players:[]})]).status,'FALLBACK_PREVIOUS_GAME');
 assert.equal(selectLineup([lineup('PREVIOUS_GAME')]).inputStatus,'FALLBACK');
});
test('agreement HIGH; lower-order disagreement MEDIUM; top-four disagreement LOW',()=>{
 assert.equal(provisionalConfidence(players,players),'HIGH');
 const minor=[...players];[minor[7],minor[8]]=[minor[8],minor[7]];
 assert.equal(provisionalConfidence(players,minor),'MEDIUM');
 const top=[...players];[top[0],top[1]]=[top[1],top[0]];
 assert.equal(provisionalConfidence(players,top),'LOW');
 assert.equal(provisionalConfidence(players,null),'LOW');
});
test('team / beat / confirmed RotoWire accepted externally; MLB final confirmation',()=>{
 for(const provider of ['TEAM','BEAT','ROTOWIRE']) assert.equal(selectLineup([lineup('ROTOWIRE'),lineup(provider,{confirmed:true})]).status,'CONFIRMED_EXTERNAL');
 assert.equal(selectLineup([lineup('ROTOWIRE'),lineup('TEAM',{confirmed:true}),lineup('MLB',{confirmed:true})]).status,'CONFIRMED_MLB');
});
test('RosterResource is audit-only and never becomes the production provisional lineup',()=>{
 const rr=lineup('ROSTERRESOURCE',{players:[...players].reverse()});
 const rw=selectLineup([rr,lineup('ROTOWIRE')]);
 assert.deepEqual(rw.players,players);
 assert.ok(rw.audit.rotowireVsRosterResource);
 const fallback=selectLineup([rr,lineup('PREVIOUS_GAME')]);
 assert.equal(fallback.status,'FALLBACK_PREVIOUS_GAME');
 assert.deepEqual(fallback.players,players);
 assert.equal(selectLineup([rr]).status,'MISSING');
});
test('credible starter conflict blocks recommendations, even if preferred source is confirmed',()=>{
 const away=side(),home=side();away.starter=selectStarter([pitcher('TEAM','A',{confirmed:true}),pitcher('MLB','B')]);
 assert.equal(away.starter.status,'CONFLICTING');
 const gate=projectionGate({away,home});assert.equal(gate.eligible,false);assert.ok(gate.reasons.includes('STARTER_CONFLICT'));
 assert.equal(recommendationGate({projectionFrozen:true,bookmaker:'x',americanOdds:100,priceTimestamp:timestamp,modelAvailable:true,exactMarketMatch:true,baseballEligibility:gate}).eligible,false);
});
test('starter change invalidates previously frozen projection; clean fresh inputs can rerun',()=>{
 const previous={away:side(),home:side()},away=side();away.starter=selectStarter([pitcher('ROTOWIRE','Starter B')]);
 const gate=projectionGate({away,home:side(),previous});assert.ok(gate.invalidations.includes('PROJECTION_INVALIDATED_STARTER_CHANGE'));assert.equal(gate.eligible,false);
 assert.equal(projectionGate({away,home:side()}).eligible,true);
});
test('confirmed lineup change invalidates frozen provisional order',()=>{
 const previous={away:side(),home:side()},away=side();away.lineup=selectLineup([lineup('MLB',{confirmed:true,players:[...players].reverse()})]);
 assert.ok(projectionGate({away,home:side(),previous}).reasons.includes('PROJECTION_INVALIDATED_LINEUP_CHANGE'));
});
test('missing starter blocks recommendations',()=>{
 const away=side();away.starter=selectStarter([]);assert.equal(away.starter.inputStatus,'MISSING');assert.equal(projectionGate({away,home:side()}).eligible,false);
});
test('recursive market information rejection',()=>{
 for(const key of ['americanOdds','sportsbook','impliedProbability','fullGameTotal','price','kelly','manualProbabilityAdjustment']) assert.throws(()=>assertBaseballOnly({nested:[{[key]:1}]}),/NON_BASEBALL_FIELD/);
 assert.throws(()=>selectLineup([lineup('ROTOWIRE',{odds:100})]));
});
test('third party failures isolated and credential-free run continues',async()=>{
 assert.equal((await safeSource('ROTOWIRE',async()=>{throw new Error('secret URL');})).status,'ROTOWIRE_UNAVAILABLE');
 const original=globalThis.fetch;globalThis.fetch=async()=>{throw new Error('down');};
 try {const x=await collectSources('2026-09-26',{});assert.equal(x.lineups.status,'ROTOWIRE_UNAVAILABLE');assert.equal(x.news.status,'MLB_NEWS_UNAVAILABLE');assert.equal(x.rr.status,'ROSTERRESOURCE_UNAVAILABLE');} finally {globalThis.fetch=original;}
});
test('source timestamps and statuses preserved; stale projection rejected',()=>{
 const row=lineup('ROTOWIRE');const result=selectLineup([row]);assert.equal(result.timestamp,row.timestamp);assert.equal(result.retrievedAt,row.retrievedAt);assert.equal(result.inputStatus,'PRIMARY');
 assert.equal(selectLineup([lineup('ROTOWIRE',{timestamp:'2020-01-01'}),lineup('PREVIOUS_GAME')]).status,'FALLBACK_PREVIOUS_GAME');
});
test('news explicit scratch overrides only with identified replacement/order, no speculation',()=>{
 const older=lineup('ROTOWIRE',{timestamp:new Date(Date.now()-3600000).toISOString()});
 const n={timestamp,source:'team',verified:true,explicit:true,action:'SCRATCH',player:players[0],position:1,replacement:'New Hitter'};
 assert.equal(selectLineup([older],[n]).players[0],'New Hitter');
 assert.equal(selectLineup([older],[{...n,verified:false}]).players[0],players[0]);
 assert.equal(selectLineup([older],[{...n,replacement:null}]).unresolved,true);
});
test('weighted audit emphasizes top four without model adjustment',()=>{
 const top=[...players];top[0]='New';const bottom=[...players];bottom[8]='New';assert.ok(lineupDelta(players,top).weightedSlotAccuracy<lineupDelta(players,bottom).weightedSlotAccuracy);
});
test('SRM request never alters any component or applies probability adjustment',()=>{
 const current={offense:1,bullpen:2,defense:3,park:4,weather:5,umpire:6,variance:7,calibration:8,thresholds:9};const saved=structuredClone(current);
 const reviews=srmReview([{recommendedAction:'SRM_REVIEW_RECOMMENDED',action:'PITCH_COUNT_LIMIT',source:'team',timestamp,reason:'60 pitches'}],current);
 assert.deepEqual(current,saved);assert.equal(reviews[0].applied,false);assert.equal(reviews[0].expectedMagnitude,null);assert.equal(reviews[0].affectedComponent,'starter quality/duration only');
});
test('RotoWire schema maps names/order/status, never provider IDs as MLB IDs or market fields',()=>{
 const payload={Date:'2026-09-26',Games:[{DateTime:timestamp,Teams:[{Code:'NYY',IsHome:1,LineupStatus:'Expected',Players:players.map((_,i)=>({Id:i,FirstName:'Player',LastName:String(i+1),BattingSpot:i+1})),StartingPitcher:{Id:777,FirstName:'Starter',LastName:'A'},Odds:-200}]}]};
 const row=normalizeRotowire(payload,payload.Date,timestamp)[0].teams[0];assert.deepEqual(row.lineup.players,players);assert.equal(row.lineup.confirmed,false);assert.equal(row.starter.id,undefined);assertBaseballOnly(row);
 payload.Games[0].Teams[0].LineupStatus='Confirmed';assert.equal(normalizeRotowire(payload,payload.Date,timestamp)[0].teams[0].lineup.confirmed,true);
 assert.throws(()=>normalizeRotowire(payload,'2026-09-25',timestamp));
});
test('news parser identifies workload for review without auto-applying narrative',()=>{
 const news=parseNews(`<rss><item><title><![CDATA[Starter A on pitch-count restriction]]></title><link>https://mlb.com/news/x</link><pubDate>${timestamp}</pubDate></item></rss>`);
 assert.equal(news[0].recommendedAction,'SRM_REVIEW_RECOMMENDED');assert.equal(news[0].explicit,false);
});
test('missing governance cannot bypass postfreeze Kelly recommendation gate',()=>{
 const result=enrichPriceDependentRecommendation({gamePk:1,segment:'full',side:'under',line:.5,bookmaker:'x',americanOdds:100,lastUpdatedAt:timestamp,modelAvailable:true,productionConditionalPct:60},{projectionFrozen:true});
 assert.equal(result.recommendationEligible,false);assert.equal(result.productionKellyPct,null);
});

test('suffix and diacritic variants do not create false lineup or starter changes',()=>{
 const previous={away:side(),home:side()};
 previous.away.lineup={...previous.away.lineup,players:['Fernando Tatis Jr.',...players.slice(1)]};
 previous.away.starter={...previous.away.starter,name:'Ronald Acuña Jr.'};
 const away=side();
 away.lineup={...away.lineup,players:['Fernando Tatis',...players.slice(1)]};
 away.starter={...away.starter,name:'Ronald Acuna'};
 const gate=projectionGate({away,home:side(),previous});
 assert.equal(gate.requiresCleanRerun,false);
 assert.equal(gate.invalidations.length,0);
 assert.equal(lineupDelta(previous.away.lineup.players,away.lineup.players).added.length,0);
 assert.equal(lineupDelta(previous.away.lineup.players,away.lineup.players).removed.length,0);
 assert.equal(lineupDelta(previous.away.lineup.players,away.lineup.players).exactSlots,9);
 assert.equal(sameMlbIdentityName('Bobby Witt Jr.','Bobby Witt'),true);
});
test('resolved MLB IDs take priority and real identity/order changes still invalidate',()=>{
 const previous={away:side(),home:side()},away=side();
 previous.away.lineup={...previous.away.lineup,resolvedMlbIds:[1,2,3,4,5,6,7,8,9]};
 away.lineup={...away.lineup,resolvedMlbIds:[1,2,3,4,5,6,7,9,8]};
 previous.away.starter={...previous.away.starter,name:'Same Name',resolvedMlbId:100};
 away.starter={...away.starter,name:'Same Name',resolvedMlbId:101};
 const gate=projectionGate({away,home:side(),previous});
 assert.ok(gate.invalidations.includes('PROJECTION_INVALIDATED_STARTER_CHANGE'));
 assert.ok(gate.invalidations.includes('PROJECTION_INVALIDATED_LINEUP_CHANGE'));
 assert.equal(gate.requiresCleanRerun,true);
});
test('starter source suffix variants do not create false source conflict',()=>{
 const s=selectStarter([pitcher('ROTOWIRE','Bobby Witt'),pitcher('MLB','Bobby Witt Jr.')]);
 assert.notEqual(s.status,'CONFLICTING');
 assert.equal(s.conflict,null);
});
