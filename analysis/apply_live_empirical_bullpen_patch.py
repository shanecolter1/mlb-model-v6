#!/usr/bin/env python3
from pathlib import Path
p=Path('index.html')
s=p.read_text()
script='<script src="/empirical-bullpen-production.js"></script>'
if script not in s:
    if '</body>' not in s: raise SystemExit('missing </body> anchor')
    s=s.replace('</body>',script+'\n</body>',1)

old1="const currentDist=runDistribution(battingLineup,currentStart,defensivePitcherId,Number(ls.outs??cp.count?.outs??0),baseMask(offense),{balls:ls.balls??cp.count?.balls??0,strikes:ls.strikes??cp.count?.strikes??0});"
new1="const currentDist=window.empiricalBullpenEngine?.currentHalfMixture?.({live,pitchingSide:fieldingSide,lineup:battingLineup,startIdx:currentStart,pitcherId:defensivePitcherId,outs:Number(ls.outs??cp.count?.outs??0),mask:baseMask(offense),currentCount:{balls:ls.balls??cp.count?.balls??0,strikes:ls.strikes??cp.count?.strikes??0}})||runDistribution(battingLineup,currentStart,defensivePitcherId,Number(ls.outs??cp.count?.outs??0),baseMask(offense),{balls:ls.balls??cp.count?.balls??0,strikes:ls.strikes??cp.count?.strikes??0});"
if old1 in s:s=s.replace(old1,new1,1)
elif new1 not in s:raise SystemExit('currentDist anchor not found')

old2="const currentAdditional=phase==='live'?runDistribution(battingLineup,currentStart,defensivePitcherId,Number(ls.outs??cp.count?.outs??0),baseMask(ls.offense||{}),{balls:ls.balls??cp.count?.balls??0,strikes:ls.strikes??cp.count?.strikes??0}):[1,0,0,0,0,0,0];"
new2="const currentAdditional=phase==='live'?(window.empiricalBullpenEngine?.currentHalfMixture?.({live,pitchingSide:fieldingSide,lineup:battingLineup,startIdx:currentStart,pitcherId:defensivePitcherId,outs:Number(ls.outs??cp.count?.outs??0),mask:baseMask(ls.offense||{}),currentCount:{balls:ls.balls??cp.count?.balls??0,strikes:ls.strikes??cp.count?.strikes??0}})||runDistribution(battingLineup,currentStart,defensivePitcherId,Number(ls.outs??cp.count?.outs??0),baseMask(ls.offense||{}),{balls:ls.balls??cp.count?.balls??0,strikes:ls.strikes??cp.count?.strikes??0})):[1,0,0,0,0,0,0];"
if old2 in s:s=s.replace(old2,new2,1)
elif new2 not in s:raise SystemExit('currentAdditional anchor not found')

p.write_text(s)
print('empirical bullpen live probability patch applied; UI markup otherwise preserved')
