(function(){
'use strict';
// Production translation of the validated live_half_inning_state_engine.
// Uses the existing frozen empirical PA transition table and batter/pitcher outcome engine.
const CAP=6, TOL=1e-12, MAX_PA=72;
function norm(a){const s=a.reduce((x,y)=>x+Number(y||0),0)||1;return a.map(x=>Number(x||0)/s)}
function eventVector(batterId,pitcherId){const p=window.outcomeProb(batterId,pitcherId,null);return {out:Number(p.out||0),bb:Number(p.bb||0),single:Number(p.single||0),double:Number(p.double||0),triple:Number(p.triple||0),hr:Number(p.hr||0)}}
function countAdjustedEventVector(base,outs,mask,count){
  // Existing dashboard count-state calibration remains the authoritative first-PA count adjustment.
  // If no event-level count model is exposed, preserve the frozen PA event vector rather than inventing one.
  if(!count)return base;
  if(typeof window.countConditionedOutcomeProb==='function'){
    try{const q=window.countConditionedOutcomeProb(base,outs,mask,count);if(q&&typeof q==='object')return q}catch(_){ }
  }
  return base;
}
function liveStateDistribution(lineup,startIdx,pitcherId,outs=0,mask=0,currentCount=null){
  if(!Array.isArray(lineup)||!lineup.length||typeof window.empiricalPATransitions!=='function'||typeof window.outcomeProb!=='function')return null;
  // sparse active-state map keyed by outs|mask|runs|batterIndex
  let active=new Map([[`${Number(outs)}|${Number(mask)}|0|${Number(startIdx)}`,1]]);
  const final=Array(CAP+1).fill(0), joint=Array.from({length:CAP+1},()=>Array(9).fill(0));
  let depth=0, first=true, unresolved=1;
  for(;depth<MAX_PA && unresolved>TOL;depth++){
    const next=new Map();
    for(const [key,mass] of active){
      if(!(mass>0))continue;
      const [o0,m0,r0,b0]=key.split('|').map(Number);
      const batterId=lineup[b0]?.id;
      let ev=eventVector(batterId,pitcherId);
      if(first)ev=countAdjustedEventVector(ev,o0,m0,currentCount);
      const ni=(b0+1)%lineup.length;
      for(const kind of ['out','bb','single','double','triple','hr']){
        const pk=Number(ev[kind]||0);if(!(pk>0))continue;
        const rows=window.empiricalPATransitions(kind,o0,m0)||[];
        for(const tr of rows){
          const w=mass*pk*Number(tr.p||0);if(!(w>0))continue;
          const no=Math.min(3,o0+Number(tr.outs_added||0));
          const rr=Math.min(CAP,r0+Number(tr.runs||0));
          if(no>=3){final[rr]+=w;joint[rr][ni]+=w;continue}
          const nm=Number(tr.post_mask||0), nk=`${no}|${nm}|${rr}|${ni}`;
          next.set(nk,(next.get(nk)||0)+w);
        }
      }
    }
    active=next;first=false;unresolved=0;for(const v of active.values())unresolved+=v;
  }
  // Numerical emergency tail: preserve mass in its current run bucket if MAX_PA is ever reached.
  if(unresolved>0){for(const [key,mass] of active){const r=Number(key.split('|')[2]);final[Math.min(CAP,r)]+=mass}}
  const dist=norm(final);
  const nextLine=Array(9).fill(0);for(let r=0;r<=CAP;r++)for(let b=0;b<9;b++)nextLine[b]+=joint[r][b];
  const rs=nextLine.reduce((a,b)=>a+b,0);if(rs>0)for(let i=0;i<9;i++)nextLine[i]/=rs;
  return {dist,nextLineupDistribution:nextLine,unresolvedProbability:Math.max(0,unresolved),remainingPaIterations:depth,converged:unresolved<=TOL};
}
const legacy=window.runDistribution;
function productionRunDistribution(lineup,startIdx,pitcherId,outs=0,mask=0,currentCount=null){
  const res=liveStateDistribution(lineup,startIdx,pitcherId,outs,mask,currentCount);
  if(!res)return legacy(lineup,startIdx,pitcherId,outs,mask,currentCount);
  return res.dist;
}
window.empiricalLiveStateEngine={liveStateDistribution,legacyRunDistribution:legacy,version:'validated-live-state-convergence-v1',unresolvedTolerance:TOL,emergencyMaxRemainingPA:MAX_PA};
window.runDistribution=productionRunDistribution;
})();
