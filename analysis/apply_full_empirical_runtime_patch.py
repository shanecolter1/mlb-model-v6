#!/usr/bin/env python3
from pathlib import Path
p=Path('index.html')
s=p.read_text(encoding='utf-8')
data='<script src="/production-data-orchestrator.js"></script>'
state='<script src="/empirical-state-production.js"></script>'
bull='<script src="/empirical-bullpen-production.js"></script>'
# Keep dashboard markup/UI untouched; enforce only production runtime script order.
for tag in (data,state):
    s=s.replace(tag+'\n','').replace(tag,'')
if bull not in s:
    raise SystemExit('bullpen production script include missing')
s=s.replace(bull,data+'\n'+state+'\n'+bull)
p.write_text(s,encoding='utf-8')
print('PASS: audited data orchestrator -> empirical state -> empirical bullpen runtime order enforced')
