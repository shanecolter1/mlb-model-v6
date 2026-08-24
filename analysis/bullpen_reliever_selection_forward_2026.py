#!/usr/bin/env python3
"""Fresh 2026 forward test for the frozen 2025 expanded reliever-selection model."""
from __future__ import annotations
import csv,io,json,shutil,urllib.request,zipfile
from datetime import datetime
from pathlib import Path
import bullpen_reliever_candidate_history as cand
import bullpen_reliever_selection_model as sel
import half_inning_scoring_gate as core
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/derived/model_calibration/bullpen_transitions';CAND_CANON=BASE/'bullpen_reliever_candidate_sets_2025.csv';CAND_2026=BASE/'bullpen_reliever_candidate_sets_2026.csv';SUM_CANON=BASE/'bullpen_reliever_candidate_summary_2025.json';SUM_2026=BASE/'bullpen_reliever_candidate_summary_2026.json';OUT=BASE/'bullpen_reliever_selection_forward_2026.json';FROZEN_C=0.03;YEAR=2026
def fetch_rows_year(year):
 url=f'https://www.retrosheet.org/downloads/plays/{year}plays.zip';req=urllib.request.Request(url,headers={'User-Agent':f'mlb-model-v6 frozen {year} bullpen validation'})
 with urllib.request.urlopen(req,timeout=120) as resp:raw=resp.read()
 out=[]
 with zipfile.ZipFile(io.BytesIO(raw)) as z:
  for n in z.namelist():
   if not n.lower().endswith('.csv'):continue
   with z.open(n) as f:
    for r in csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig',newline='')):
     if not core.is_regular(r.get('gametype')):continue
     gid=str(r.get('gid') or r.get('game_id') or '').strip()
     if not gid:continue
     digits=''.join(c for c in gid if c.isdigit())
     if len(digits)<8:continue
     r['_gid']=gid;r['_day']=datetime.strptime(digits[:8],'%Y%m%d').date().toordinal();out.append(r)
 out.sort(key=lambda r:(r['_day'],r['_gid']));return out
def team_map_2026():
 data=cand.get_json(f'https://statsapi.mlb.com/api/v1/teams?sportId=1&season={YEAR}');out={}
 for t in data.get('teams',[]):
  tid=int(t['id'])
  for k in ('teamCode','fileCode','abbreviation'):
   v=cand.clean(t.get(k)).upper()
   if v:out[v]=tid
 return out
def fetch_active_pitchers_2026(team_id,day_iso,mlb_to_retro):
 data=cand.get_json(f'https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active&season={YEAR}&date={day_iso}&hydrate=person');candidates=[];unmapped=0
 for x in data.get('roster',[]):
  pos=x.get('position') or {};person=x.get('person') or {};abbr=cand.clean(pos.get('abbreviation')).upper();ptype=cand.clean(pos.get('type')).lower()
  if abbr!='P' and ptype!='pitcher':continue
  try:mid=int(person['id'])
  except Exception:continue
  retro=mlb_to_retro.get(mid)
  if retro:candidates.append((retro,mid,cand.clean((person.get('pitchHand') or {}).get('code')).upper() or None))
  else:unmapped+=1
 return candidates,unmapped,len(data.get('roster',[]))
def main():
 if not CAND_CANON.exists():raise SystemExit('2025 candidate artifact must be built before the forward test')
 groups_2025=sel.load();train_2025,_,_=sel.split(groups_2025)
 if len(train_2025)<1000:raise SystemExit(f'insufficient frozen 2025 training choice sets: {len(train_2025)}')
 X,y=sel.flat(train_2025,sel.EXP_NUM,sel.EXP_CAT);model=sel.make_model(FROZEN_C,sel.EXP_NUM,sel.EXP_CAT);model.fit(X,y)
 orig_rows=core.fetch_rows_2025;orig_team=cand.team_map;orig_roster=cand.fetch_active_pitchers
 try:
  core.fetch_rows_2025=lambda:fetch_rows_year(YEAR);cand.team_map=team_map_2026;cand.fetch_active_pitchers=fetch_active_pitchers_2026;cand.main()
 finally:
  core.fetch_rows_2025=orig_rows;cand.team_map=orig_team;cand.fetch_active_pitchers=orig_roster
 shutil.copy2(CAND_CANON,CAND_2026);shutil.copy2(SUM_CANON,SUM_2026);candidate_summary=json.loads(SUM_2026.read_text());groups_2026=sel.load()
 if len(groups_2026)<500:raise SystemExit(f'insufficient 2026 forward choice sets: {len(groups_2026)}')
 metrics=sel.evalm(model,groups_2026,sel.EXP_NUM,sel.EXP_CAT);gate={'candidate_coverage_ge_0_97':candidate_summary['actual_reliever_candidate_coverage']>=.97,'beats_uniform_logloss':metrics['log_loss']<metrics['uniform_log_loss'],'beats_prior_usage_logloss':metrics['log_loss']<metrics['prior_usage_log_loss'],'beats_prior_usage_top1':metrics['top1_accuracy']>metrics['prior_usage_top1_accuracy'],'p99_lt_1000ms':metrics['p99_inference_ms']<1000}
 rep={'status':'FRESH_2026_FORWARD_TEST','market_blind':True,'frozen_before_2026':{'training':'2025 through 2025-07-31','C':FROZEN_C,'features':sel.EXP_NUM+sel.EXP_CAT,'excluded_features':['team identity','pitcher identity','sportsbook/market inputs','future current-game usage']},'candidate_summary':candidate_summary,'forward_metrics':metrics,'promotion_gates':gate,'forward_gate_status':'PASS' if all(gate.values()) else 'BLOCKED','next_gate':'downstream inning/remaining-game run-distribution improvement; forward reliever-choice pass alone does not authorize production promotion'}
 OUT.write_text(json.dumps(rep,indent=2));print(json.dumps(rep,indent=2))
 if not all(gate.values()):raise SystemExit('2026 reliever-selection forward gate blocked')
if __name__=='__main__':main()
