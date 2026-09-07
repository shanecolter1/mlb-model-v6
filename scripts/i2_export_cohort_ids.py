from pathlib import Path
import textwrap

p=Path('.github/workflows/i2-rotation-series-edge.yml')
t=p.read_text()
marker="          python - <<'PY'\n"
s=t.index(marker)+len(marker)
e=t.index("\n          PY",s)
code=textwrap.dedent(t[s:e])
extra=r'''
print('\nCOHORT_ID_EXPORT')
print('series_game,rotation_tier,rest_bucket,gid,date,day,under,a_rank,h_rank,rest_total')
for sg in [1,2,3,4]:
    for tlabel,tpred in [('both_top3',lambda g:g['both_top3']),('any_back',lambda g:g['any_back'])]:
        for rlabel,rpred in [('0-1',lambda g:g.get('rest_total') is not None and g['rest_total']<=1),('2+',lambda g:g.get('rest_total') is not None and g['rest_total']>=2)]:
            rr=[g for g in valid if g.get('sg')==sg and tpred(g) and rpred(g)]
            for g in rr:
                print(f"{sg},{tlabel},{rlabel},{g['gid']},{g['date']},{g['day']},{int(g['under'])},{g['away_qrank']},{g['home_qrank']},{g['rest_total']}")
print('\nSUNDAY_G4_COHORT_ID_EXPORT')
print('rotation_tier,rest_bucket,gid,date,under,a_rank,h_rank,rest_total')
for tlabel,tpred in [('both_top3',lambda g:g['both_top3']),('any_back',lambda g:g['any_back'])]:
    for rlabel,rpred in [('0-1',lambda g:g.get('rest_total') is not None and g['rest_total']<=1),('2+',lambda g:g.get('rest_total') is not None and g['rest_total']>=2)]:
        rr=[g for g in valid if g.get('sg')==4 and g.get('day')=='Sunday' and tpred(g) and rpred(g)]
        for g in rr:
            print(f"{tlabel},{rlabel},{g['gid']},{g['date']},{int(g['under'])},{g['away_qrank']},{g['home_qrank']},{g['rest_total']}")
'''
exec(compile(code+'\n'+extra,'<cohort-export>','exec'),globals(),globals())
