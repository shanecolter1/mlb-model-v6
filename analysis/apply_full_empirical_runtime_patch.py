#!/usr/bin/env python3
from pathlib import Path

p = Path('index.html')
s = p.read_text(encoding='utf-8')

orchestrator = '<script src="/production-data-orchestrator.js"></script>'
park = '<script src="/park-display.js"></script>'
state = '<script src="/empirical-state-production.js"></script>'
bull = '<script src="/empirical-bullpen-production.js"></script>'

# Restore the original single-controller V11.13 polling architecture.
# The empirical prediction modules remain downstream consumers of the canonical
# game state; no external script is allowed to replace pollGame/refreshSchedule.
s = s.replace(orchestrator + '\n', '').replace(orchestrator, '')

# Directly harden the proven inline poller against observed ~2s+ Netlify/MLB
# round trips. This changes only request tolerance, not cadence or controller ownership.
s = s.replace('const LIVE_TIMEOUT_MS=1800;', 'const LIVE_TIMEOUT_MS=4000;')

# Keep display-only park context and empirical prediction modules in deterministic
# order, with the park module outside prediction math.
for tag in (park, state):
    s = s.replace(tag + '\n', '').replace(tag, '')
if bull not in s:
    raise SystemExit('bullpen production script include missing')
s = s.replace(bull, park + '\n' + state + '\n' + bull)

# Guardrail: production must have one pollGame implementation only. The removed
# orchestrator was the second runtime owner that caused controller conflicts.
if '/production-data-orchestrator.js' in s:
    raise SystemExit('production data orchestrator still included')
if 'const LIVE_TIMEOUT_MS=4000;' not in s:
    raise SystemExit('live timeout hardening missing')

p.write_text(s, encoding='utf-8')
print('PASS: restored single V11.13 controller + 4s request tolerance + display/empirical runtime order')
