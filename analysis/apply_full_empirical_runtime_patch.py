#!/usr/bin/env python3
from pathlib import Path
p=Path('index.html')
s=p.read_text(encoding='utf-8')
state='<script src="/empirical-state-production.js"></script>'
bull='<script src="/empirical-bullpen-production.js"></script>'
# Keep dashboard markup/UI untouched; only runtime script order changes.
s=s.replace(state+'\n','').replace(state,'')
if bull not in s:
    raise SystemExit('bullpen production script include missing')
s=s.replace(bull,state+'\n'+bull)
p.write_text(s,encoding='utf-8')
print('PASS: full empirical state runtime inserted before bullpen runtime')
