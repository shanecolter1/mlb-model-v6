(function(){
'use strict';
window.MLB_DATA_ORCHESTRATOR_VERSION='audited-hybrid-polling-v1';
const LIVE_REQUEST_TIMEOUT_MS=4000;
const ACTIVE_REQUEST_MS=800;
const BETWEEN_BATTERS_REQUEST_MS=1200;
const BREAK_REQUEST_MS=2200;
const TURNOVER_REQUEST_MS=750;
const FAILURE_MAX_DELAY_MS=2500;
const PREGAME_ROSTER_WINDOW_MS=12*60*60*1000;
const PREGAME_ROSTER_REFRESH_MS=60*60*1000;
const rosterLoadedAt=new Map();
const validatedSnapshots=new Map();
const pregameFingerprints=new Map();

function gameStartMs(g){return new Date(g?.gameDate||g?.gameData?.datetime?.dateTime||0).getTime()}
function sideTeamId(g,side){return g?.teams?.[side]?.team?.id||g?.gameData?.teams?.[side]?.id||null}
function probableId(g,side){return g?.gameData?.probablePitchers?.[side]?.id||g?.teams?.[side]?.probablePitcher?.id||null}
function currentMatchup(g){return g?.liveData?.plays?.currentPlay?.matchup||{}}
function liveCore(g){
  const ls=g?.liveData?.linescore||{}, cp=g?.liveData?.plays?.currentPlay||{}, m=cp.matchup||{};
  return {
    inning:Number(ls.currentInning||0),
    outs:Number(ls.outs??cp.count?.outs),
    balls:Number(ls.balls??cp.count?.balls),
    strikes:Number(ls.strikes??cp.count?.strikes),
    batterId:m.batter?.id||null,
    pitcherId:m.pitcher?.id||null,
    bases:{first:!!ls.offense?.first,second:!!ls.offense?.second,third:!!ls.offense?.third}
  };
}
function statPresent(id,group){return !!(id&&seasonStats?.[String(id)]?.[group]&&Object.keys(seasonStats[String(id)][group]).length)}
function modelInputQuality(g){
  const state=groupState(gameState(g));
  const box=g?.liveData?.boxscore||{};
  const awayLine=activeLineup(box,'away'),homeLine=activeLineup(box,'home');
  if(state==='live'){
    const c=liveCore(g);
    const validState=c.inning>0&&Number.isFinite(c.outs)&&c.outs>=0&&c.outs<=3;
    const matchup=!!c.batterId&&!!c.pitcherId;
    const fullLines=awayLine.length===9&&homeLine.length===9;
    const profiles=statPresent(c.batterId,'hitting')&&statPresent(c.pitcherId,'pitching');
    if(validState&&matchup&&fullLines&&profiles)return {level:'FULL',reason:'validated live state + matchup + lineups + player profiles'};
    if(validState&&matchup)return {level:'DEGRADED',reason:'live state valid; secondary lineup/profile context incomplete'};
    return {level:'PRIOR',reason:'critical live batter/pitcher/state context incomplete'};
  }
  const ap=probableId(g,'away'),hp=probableId(g,'home');
  const fullLines=awayLine.length===9&&homeLine.length===9;
  const starterStats=statPresent(ap,'pitching')&&statPresent(hp,'pitching');
  if(fullLines&&ap&&hp&&starterStats)return {level:'FULL',reason:'official lineups + starters + starter profiles cached'};
  if(ap||hp||awayLine.length||homeLine.length)return {level:'DEGRADED',reason:'pregame context partially available'};
  return {level:'PRIOR',reason:'pregame matchup context not yet available'};
}
window.modelInputQuality=modelInputQuality;
window.validatedGameSnapshots=validatedSnapshots;

function cacheSnapshot(g){
  const key=String(g?.gamePk||'');if(!key)return;
  const q=modelInputQuality(g);
  validatedSnapshots.set(key,{capturedAt:Date.now(),signature:gameSignature(g),quality:q,core:liveCore(g)});
}

async function preloadAllActiveRosterContext(force=false){
  const now=Date.now(), date=dateInput.value||centralClock().date;
  const targets=(cache||[]).filter(g=>{
    const state=groupState(gameState(g));if(state==='final')return false;
    const start=gameStartMs(g);if(state==='live')return true;
    return Number.isFinite(start)&&start-now<=PREGAME_ROSTER_WINDOW_MS&&start-now>=-4*60*60*1000;
  });
  const ids=[];
  for(const g of targets)for(const side of ['away','home']){
    const id=sideTeamId(g,side);if(!id)continue;
    const key=String(id),fresh=now-(rosterLoadedAt.get(key)||0)<PREGAME_ROSTER_REFRESH_MS;
    if(force||!rosterCache[key]||!fresh)ids.push(key);
  }
  const unique=[...new Set(ids)];
  if(unique.length){
    try{
      const data=await getJSON(`${API}?type=rosters&teamIds=${encodeURIComponent(unique.join(','))}&date=${encodeURIComponent(date)}`,{timeoutMs:8000});
      for(const [id,value] of Object.entries(data.teams||{})){
        if(value?.roster?.length){rosterCache[id]=value.roster;rosterLoadedAt.set(String(id),Date.now())}
      }
      markSource('rosters',true);persistDailyCache();
    }catch(e){markSource('rosters',false,e);console.warn('Pregame active-roster preload failed',e)}
  }
  const season=String(new Date(`${date}T12:00:00`).getFullYear());
  try{await loadSeasonStats(targets,season,force);markSource('stats',true)}catch(e){markSource('stats',false,e)}
  try{
    if(force||Date.now()-Number(savantFetchMeta?.leaderboards||0)>=6*60*60*1000){
      await loadSavantLeaderboards(season,force);markSource('savant',true);
    }
  }catch(e){markSource('savant',false,e)}
  persistDailyCache();
  return targets.length;
}
window.preloadAllActiveRosterContext=preloadAllActiveRosterContext;

function fingerprint(g){
  const box=g?.liveData?.boxscore||{};
  return JSON.stringify({
    awayStarter:probableId(g,'away'),homeStarter:probableId(g,'home'),
    awayLine:activeLineup(box,'away').map(x=>x.id),homeLine:activeLineup(box,'home').map(x=>x.id)
  });
}
function detectPregameInvalidation(){
  let changed=false;
  for(const g of cache||[]){
    if(groupState(gameState(g))!=='scheduled')continue;
    const key=String(g.gamePk),next=fingerprint(g),prior=pregameFingerprints.get(key);
    if(prior&&prior!==next)changed=true;
    pregameFingerprints.set(key,next);
  }
  return changed;
}

const originalPrepareContext=prepareContext;
prepareContext=async function(force=false){
  await preloadAllActiveRosterContext(force);
  return originalPrepareContext(force);
};

const originalRefreshSchedule=refreshSchedule;
refreshSchedule=async function(force=false){
  const out=await originalRefreshSchedule(force);
  const invalidated=detectPregameInvalidation();
  if(invalidated)preloadAllActiveRosterContext(true).then(()=>render(true));
  (cache||[]).forEach(cacheSnapshot);
  return out;
};

nextGamePollDelay=function(g){
  const state=gameState(g),cp=g?.liveData?.plays?.currentPlay||{};
  if(['Final','Game Over','Completed Early'].includes(state))return null;
  if(/delay|postpon/i.test(state))return 20000;
  if(immediateHalfRollover(g).active)return TURNOVER_REQUEST_MS;
  if(cp.about?.isComplete===false)return ACTIVE_REQUEST_MS;
  const abstract=g?.gameData?.status?.abstractGameState||g?.status?.abstractGameState;
  if(abstract==='Live')return BETWEEN_BATTERS_REQUEST_MS;
  return BREAK_REQUEST_MS;
};

pollGame=async function(gamePk){
  const key=String(gamePk),c=gameControllers.get(key);if(!c||!autoEl.checked)return;
  c.timer=null;c.inFlight=true;c.lastPollStartedAt=Date.now();
  const base=cache.find(g=>String(g.gamePk)===key);if(!base){stopGameController(key);return}
  const drift=Math.max(0,Date.now()-(c.expectedRunAt||Date.now()));c.timerDriftMs=drift;
  c.abortController?.abort();const ac=new AbortController();c.abortController=ac;const seq=++c.sequence;const started=performance.now();
  const timeout=setTimeout(()=>ac.abort(),LIVE_REQUEST_TIMEOUT_MS);
  try{
    const detail=await getJSON(`${API}?type=feed&gamePk=${encodeURIComponent(gamePk)}&_=${Date.now()}`,{timeoutMs:LIVE_REQUEST_TIMEOUT_MS,signal:ac.signal});
    if(seq!==c.sequence||gameControllers.get(key)!==c)return;
    const updated=mergeGame(base,detail),tier=classifyEventChange(base,updated);
    cache=cache.map(g=>String(g.gamePk)===key?updated:g);
    cacheSnapshot(updated);
    updateDynamicPitcherState(updated);
    refreshSavantGame(gamePk,false).then(changed=>{if(changed){updateDynamicPitcherState(updated);cacheSnapshot(updated);render(false,new Set([key]))}});
    const latencyMs=performance.now()-started,previous=feedHealth.get(key)||{},sourceDataAt=mlbStateTimestamp(detail)||Date.now();
    feedHealth.set(key,{...previous,lastSuccessAt:Date.now(),sourceDataAt,latencyMs,failures:0,tier,pollMs:c.currentInterval,timerDriftMs:drift,backgroundThrottled:drift>2500,quality:modelInputQuality(updated).level});
    c.failures=0;lastFeedReceivedAt=new Date();
    if(tier==='structural'){
      const season=String(new Date(dateInput.value+'T12:00:00').getFullYear());
      loadSeasonStats([updated],season,false).then(changed=>{if(changed){cacheSnapshot(updated);render(false,new Set([key]))}});
    }
    if(tier!=='none'||Date.now()-(c.lastHealthRender||0)>2000){render(false,new Set([key]));c.lastHealthRender=Date.now()}else{updateMetrics();updateHealthDisplay(key)}
    const state=gameState(updated);
    if(groupState(state)!=='live'){
      if(groupState(state)==='final')queuePostgameCloseout(updated);
      stopGameController(key);scheduleScheduleRefresh(0);runOvernightScheduler(false);return;
    }
    c.currentInterval=nextGamePollDelay(updated)||BREAK_REQUEST_MS;
  }catch(e){
    if(gameControllers.get(key)!==c)return;
    if(e.name!=='AbortError')console.warn(`Game ${gamePk} live feed`,e);
    c.failures=(c.failures||0)+1;
    feedHealth.set(key,{...(feedHealth.get(key)||{}),failures:c.failures,lastErrorAt:Date.now(),quality:modelInputQuality(base).level});
    c.currentInterval=Math.min(FAILURE_MAX_DELAY_MS,ACTIVE_REQUEST_MS*Math.pow(1.35,Math.min(c.failures,5)));
  }finally{
    clearTimeout(timeout);
    const current=gameControllers.get(key);
    if(current===c){c.inFlight=false;c.abortController=null;scheduleGamePoll(key,c.currentInterval||ACTIVE_REQUEST_MS)}
    const now=new Date();document.getElementById('mRefresh').textContent=now.toLocaleTimeString([],{hour:'numeric',minute:'2-digit',second:'2-digit'});
  }
};

const originalGameCard=gameCard;
gameCard=function(g){
  let html=originalGameCard(g);const q=modelInputQuality(g);
  const cls=q.level==='FULL'?'live':q.level==='DEGRADED'?'delayed':'disconnected';
  const badge=`<span class="health ${cls}" title="${esc(q.reason)}"><span class="health-dot"></span><span class="health-copy">MODEL ${q.level}</span></span>`;
  html=html.replace('<div class="footer">',`<div class="footer">${badge}`);
  return html;
};

// The inline dashboard starts its first refresh before external scripts load. Reconcile
// immediately after installing the audited orchestrator, then restart live controllers
// so all subsequent live requests use this transport policy.
setTimeout(async()=>{
  try{
    await preloadAllActiveRosterContext(false);
    (cache||[]).forEach(cacheSnapshot);
    if(typeof restartAllLiveGameControllers==='function')restartAllLiveGameControllers('audited-data-orchestrator');
    render(true);
  }catch(e){console.warn('Audited data orchestrator initialization',e)}
},0);
})();
