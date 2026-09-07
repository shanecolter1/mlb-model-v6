#!/usr/bin/env python3
import csv, gzip, math, statistics, sys
from collections import defaultdict
from pathlib import Path

if len(sys.argv) != 3:
    raise SystemExit('usage: i2_market_pricing_16cell.py <cohort_stdout.txt> <master.csv.gz>')
cohort_path = Path(sys.argv[1])
master_path = Path(sys.argv[2])
outdir = Path('analysis_outputs')
outdir.mkdir(exist_ok=True)


def fnum(v):
    try:
        if v is None or str(v).strip() == '':
            return None
        return float(v)
    except Exception:
        return None


def implied_american(o):
    o=fnum(o)
    if o is None or o == 0:
        return None
    return (-o)/((-o)+100.0) if o < 0 else 100.0/(o+100.0)


def devig_under(over_o, under_o):
    po=implied_american(over_o); pu=implied_american(under_o)
    if po is None or pu is None or po+pu <= 0:
        return None
    return pu/(po+pu)


def fair_american(p):
    if p is None or p <= 0 or p >= 1:
        return None
    return -100.0*p/(1.0-p) if p >= .5 else 100.0*(1.0-p)/p


def quantile(vals, q):
    vals=sorted(vals)
    n=len(vals)
    if not n: return None
    if n == 1: return vals[0]
    h=(n-1)*q
    lo=int(math.floor(h)); hi=int(math.ceil(h))
    if lo == hi: return vals[lo]
    return vals[lo] + (vals[hi]-vals[lo])*(h-lo)


def mean(vals):
    vals=[v for v in vals if v is not None]
    return sum(vals)/len(vals) if vals else None


def pct(v):
    return None if v is None else 100.0*v

# Parse exact cohort rows produced by the frozen exporter.
lines=cohort_path.read_text(encoding='utf-8').splitlines()
try:
    i0=lines.index('COHORT_ID_EXPORT')
    i1=lines.index('SUNDAY_G4_COHORT_ID_EXPORT')
except ValueError as e:
    raise SystemExit(f'cohort markers missing: {e}')

main_text='\n'.join(lines[i0+1:i1]).strip()
sun_text='\n'.join(lines[i1+1:]).strip()
main_rows=list(csv.DictReader(main_text.splitlines()))
sun_rows=list(csv.DictReader(sun_text.splitlines()))
for r in main_rows:
    r['series_game']=int(r['series_game'])
    r['under']=int(r['under'])
for r in sun_rows:
    r['under']=int(r['under'])

# Load canonical released market master.
master={}
with gzip.open(master_path,'rt',encoding='utf-8-sig',newline='') as f:
    rd=csv.DictReader(f)
    required={'retro_game_id','dk_total_open_total','dk_total_open_overOdds','dk_total_open_underOdds','inning2_total_runs'}
    missing=required-set(rd.fieldnames or [])
    if missing:
        raise SystemExit(f'missing master columns: {sorted(missing)}')
    for r in rd:
        gid=r['retro_game_id']
        total=fnum(r['dk_total_open_total'])
        i2=fnum(r['inning2_total_runs'])
        y=None if i2 is None else (1 if i2 == 0 else 0)
        x=devig_under(r['dk_total_open_overOdds'],r['dk_total_open_underOdds'])
        master[gid]={
            'gid':gid,
            'total':total,
            'over_odds':fnum(r['dk_total_open_overOdds']),
            'under_odds':fnum(r['dk_total_open_underOdds']),
            'devig_under':x,
            'under':y,
            'game_date':r.get('game_date',''),
        }

# Exact-total empirical I2 Under baselines.
by_total=defaultdict(list)
by_total_x=defaultdict(list)
for r in master.values():
    if r['total'] is not None and r['under'] is not None:
        by_total[r['total']].append(r['under'])
        if r['devig_under'] is not None:
            by_total_x[r['total']].append((r['devig_under'],r['under']))
base_total={t:sum(v)/len(v) for t,v in by_total.items() if v}

# Total-fixed-effect linear juice adjustment: y = alpha_total + beta*(devig_under - mean_devig_total).
# This is only a pricing proxy; total-only residual remains the primary promotion statistic.
means_x={t:mean([x for x,y in vals]) for t,vals in by_total_x.items() if vals}
means_y={t:mean([y for x,y in vals]) for t,vals in by_total_x.items() if vals}
num=den=0.0
for t,vals in by_total_x.items():
    mx=means_x[t]; my=means_y[t]
    if mx is None or my is None: continue
    for x,y in vals:
        dx=x-mx
        num += dx*(y-my)
        den += dx*dx
juice_beta = num/den if den > 0 else None

def juice_pred(r):
    t=r['total']; x=r['devig_under']
    if juice_beta is None or t not in base_total or x is None or t not in means_x:
        return None
    p=base_total[t] + juice_beta*(x-means_x[t])
    return min(.99,max(.01,p))


def interpret(resid):
    if resid is None: return 'Insufficient market coverage'
    if resid >= 5: return 'Large residual: not fully priced'
    if resid >= 2: return 'Positive residual: partially priced'
    if resid > -2: return 'Small residual: largely priced'
    return 'Negative residual: market baseline >= cohort'


def summarize(rows):
    n=len(rows)
    under_rate=sum(int(r['under']) for r in rows)/n if n else None
    mrows=[master.get(r['gid']) for r in rows]
    mrows=[m for m in mrows if m is not None and m['total'] is not None]
    covered_n=len(mrows)
    covered_under=mean([m['under'] for m in mrows])
    totals=[m['total'] for m in mrows]
    baselines=[base_total.get(m['total']) for m in mrows if base_total.get(m['total']) is not None]
    total_baseline=mean(baselines)
    residual_all=(under_rate-total_baseline)*100 if under_rate is not None and total_baseline is not None else None
    residual_cov=(covered_under-total_baseline)*100 if covered_under is not None and total_baseline is not None else None
    jrows=[m for m in mrows if m['devig_under'] is not None]
    jpred=[juice_pred(m) for m in jrows]
    jpred=[p for p in jpred if p is not None]
    juice_base=mean(jpred)
    juice_resid_cov=(covered_under-juice_base)*100 if covered_under is not None and juice_base is not None else None
    return {
        'cohort_n':n,
        'market_covered_n':covered_n,
        'market_coverage_pct':100*covered_n/n if n else None,
        'historical_i2_under_pct':pct(under_rate),
        'market_covered_i2_under_pct':pct(covered_under),
        'fair_under_american':fair_american(under_rate),
        'q1_total':quantile(totals,.25),
        'median_total':quantile(totals,.5),
        'q3_total':quantile(totals,.75),
        'share_total_le_7_5_pct':100*sum(t<=7.5 for t in totals)/covered_n if covered_n else None,
        'share_total_eq_8_0_pct':100*sum(t==8.0 for t in totals)/covered_n if covered_n else None,
        'share_total_eq_8_5_pct':100*sum(t==8.5 for t in totals)/covered_n if covered_n else None,
        'share_total_ge_9_0_pct':100*sum(t>=9.0 for t in totals)/covered_n if covered_n else None,
        'median_devig_fullgame_under_pct':100*quantile([m['devig_under'] for m in jrows],.5) if jrows else None,
        'juice_covered_n':len(jrows),
        'total_only_market_baseline_i2_under_pct':pct(total_baseline),
        'residual_all_vs_total_pp':residual_all,
        'residual_covered_vs_total_pp':residual_cov,
        'juice_enhanced_market_baseline_i2_under_pct':pct(juice_base),
        'residual_covered_vs_total_plus_juice_pp':juice_resid_cov,
        'interpretation':interpret(residual_cov),
    }

# 16-cell table.
keys=[]
for sg in [1,2,3,4]:
    for tier in ['both_top3','any_back']:
        for rest in ['0-1','2+']:
            keys.append((sg,tier,rest))

bykey=defaultdict(list)
for r in main_rows:
    bykey[(r['series_game'],r['rotation_tier'],r['rest_bucket'])].append(r)

summary_rows=[]
for sg,tier,rest in keys:
    s=summarize(bykey[(sg,tier,rest)])
    summary_rows.append({'series_game':sg,'rotation_tier':tier,'rest_bucket':rest,**s})

fieldnames=list(summary_rows[0].keys())
with open(outdir/'i2_market_pricing_16cell.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader(); w.writerows(summary_rows)

# Focused G4 versus G1-G3 pooled comparison within identical pitching/rest strata.
focus=[]
for tier in ['both_top3','any_back']:
    for rest in ['0-1','2+']:
        g4=bykey[(4,tier,rest)]
        early=[]
        for sg in [1,2,3]: early += bykey[(sg,tier,rest)]
        a=summarize(early); b=summarize(g4)
        row={
            'rotation_tier':tier,'rest_bucket':rest,
            'g1_g3_n':a['cohort_n'],'g1_g3_under_pct':a['historical_i2_under_pct'],'g1_g3_median_total':a['median_total'],
            'g1_g3_total_baseline_pct':a['total_only_market_baseline_i2_under_pct'],'g1_g3_residual_covered_pp':a['residual_covered_vs_total_pp'],
            'g4_n':b['cohort_n'],'g4_under_pct':b['historical_i2_under_pct'],'g4_median_total':b['median_total'],
            'g4_total_baseline_pct':b['total_only_market_baseline_i2_under_pct'],'g4_residual_covered_pp':b['residual_covered_vs_total_pp'],
            'g4_minus_g1_g3_under_pp':None if a['historical_i2_under_pct'] is None or b['historical_i2_under_pct'] is None else b['historical_i2_under_pct']-a['historical_i2_under_pct'],
            'g4_minus_g1_g3_median_total':None if a['median_total'] is None or b['median_total'] is None else b['median_total']-a['median_total'],
            'g4_minus_g1_g3_market_baseline_pp':None if a['total_only_market_baseline_i2_under_pct'] is None or b['total_only_market_baseline_i2_under_pct'] is None else b['total_only_market_baseline_i2_under_pct']-a['total_only_market_baseline_i2_under_pct'],
            'g4_minus_g1_g3_residual_pp':None if a['residual_covered_vs_total_pp'] is None or b['residual_covered_vs_total_pp'] is None else b['residual_covered_vs_total_pp']-a['residual_covered_vs_total_pp'],
        }
        row['market_pricing_read']='Market did not move enough' if (row['g4_minus_g1_g3_under_pp'] is not None and row['g4_minus_g1_g3_market_baseline_pp'] is not None and row['g4_minus_g1_g3_under_pp']-row['g4_minus_g1_g3_market_baseline_pp']>=3) else 'Market movement broadly explains lift'
        focus.append(row)
with open(outdir/'i2_market_pricing_game4_focus.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(focus[0].keys())); w.writeheader(); w.writerows(focus)

# Sunday G4 exact four cells.
sby=defaultdict(list)
for r in sun_rows:
    sby[(r['rotation_tier'],r['rest_bucket'])].append(r)
sunday=[]
for tier in ['both_top3','any_back']:
    for rest in ['0-1','2+']:
        s=summarize(sby[(tier,rest)])
        sunday.append({'rotation_tier':tier,'rest_bucket':rest,**s})
with open(outdir/'i2_market_pricing_sunday_g4.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(sunday[0].keys())); w.writeheader(); w.writerows(sunday)

# Validation and audit notes.
checks=[]
for t in [6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0,10.5,11.0]:
    vals=by_total.get(t,[])
    checks.append((t,len(vals),100*base_total[t] if t in base_total else None))
with open(outdir/'i2_market_pricing_validation.txt','w',encoding='utf-8') as f:
    f.write(f'master_rows={len(master)}\n')
    f.write(f'cohort_rows={len(main_rows)}\n')
    f.write(f'sunday_g4_rows={len(sun_rows)}\n')
    f.write(f'juice_fixed_effect_beta={juice_beta}\n')
    f.write('archive_caveat=DK market archive is partial through 2025-08-16; market_covered_n is reported separately from cohort_n.\n')
    f.write('pricing_rule=Pregame total is a market-pricing benchmark, not a causal control. Total-only residual is primary; juice-enhanced residual is secondary.\n')
    f.write('total,n,i2_under_pct\n')
    for t,n,p in checks: f.write(f'{t},{n},{p}\n')

print('WROTE', outdir/'i2_market_pricing_16cell.csv')
print('WROTE', outdir/'i2_market_pricing_game4_focus.csv')
print('WROTE', outdir/'i2_market_pricing_sunday_g4.csv')
print('WROTE', outdir/'i2_market_pricing_validation.txt')
