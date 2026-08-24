#!/usr/bin/env python3
from pathlib import Path
p=Path('index.html')
s=p.read_text()
old="""function currentPitcherId(live, side){
  const box=live?.boxscore||{};
  const team=box?.teams?.[side];
  return team?.pitchers?.slice(-1)[0] || null;
}
function activeLineup(box, side){
  const players=Object.values(box?.teams?.[side]?.players||{});
  const bySlot=new Map();
  players.forEach(p=>{
    const raw=Number(p?.battingOrder||0);
    const slot=Math.floor(raw/100);
    if(slot>=1 && slot<=9){
      const existing=bySlot.get(slot);
      if(!existing || raw >= Number(existing.battingOrder||0)) bySlot.set(slot,p);
    }
  });
  return [...bySlot.entries()].sort((a,b)=>a[0]-b[0]).map(([slot,p])=>({
    id:p.person?.id, name:p.person?.fullName||'—', slot
  }));
}
"""
new="""function currentPitcherId(live, side){
  const box=live?.boxscore||{};
  const team=box?.teams?.[side];
  const listed=team?.pitchers?.slice(-1)[0];
  if(listed) return listed;
  // MLB can mark a game Live before the boxscore pitcher array is populated.
  // currentPlay is authoritative for the team currently on defense.
  const ls=live?.linescore||{}, cp=live?.plays?.currentPlay||{};
  const fieldingSide=ls.isTopInning===false?'away':'home';
  if(side===fieldingSide && cp?.matchup?.pitcher?.id) return cp.matchup.pitcher.id;
  return null;
}
function activeLineup(box, side){
  const team=box?.teams?.[side]||{};
  const players=Object.values(team.players||{});
  const bySlot=new Map();
  players.forEach(p=>{
    const raw=Number(p?.battingOrder||0);
    const slot=Math.floor(raw/100);
    if(slot>=1 && slot<=9){
      const existing=bySlot.get(slot);
      if(!existing || raw >= Number(existing.battingOrder||0)) bySlot.set(slot,p);
    }
  });
  if(bySlot.size){
    return [...bySlot.entries()].sort((a,b)=>a[0]-b[0]).map(([slot,p])=>({id:p.person?.id,name:p.person?.fullName||'—',slot}));
  }
  // Early live feeds sometimes expose the official order as IDs before each player
  // object receives battingOrder. Use those authoritative team arrays as fallback.
  const ids=(team.battingOrder||team.batters||[]).map(Number).filter(Boolean).slice(0,9);
  return ids.map((id,i)=>{
    const p=team.players?.['ID'+id]||{};
    return {id,name:p.person?.fullName||p.person?.name||'—',slot:i+1};
  });
}
"""
if old not in s: raise SystemExit('pitcher/lineup target not found')
s=s.replace(old,new,1)
old2="  const offensePitcherId=currentPitcherId(live,battingSide);"
new2="  const offensePitcherId=currentPitcherId(live,battingSide)||gd.probablePitchers?.[battingSide]?.id||g.teams?.[battingSide]?.probablePitcher?.id||null;"
if old2 not in s: raise SystemExit('offense pitcher target not found')
s=s.replace(old2,new2,1)
old3="const pid=currentPitcherId(live,'away');return compactTeamPanel"
new3="const pid=currentPitcherId(live,'away')||gd.probablePitchers?.away?.id||g.teams?.away?.probablePitcher?.id;return compactTeamPanel"
if old3 not in s: raise SystemExit('away panel target not found')
s=s.replace(old3,new3,1)
old4="const pid=currentPitcherId(live,'home');return compactTeamPanel"
new4="const pid=currentPitcherId(live,'home')||gd.probablePitchers?.home?.id||g.teams?.home?.probablePitcher?.id;return compactTeamPanel"
if old4 not in s: raise SystemExit('home panel target not found')
s=s.replace(old4,new4,1)
p.write_text(s)
print('patched live context fallbacks')
