from pathlib import Path
import textwrap

p = Path('.github/workflows/i2-rotation-series-edge.yml')
t = p.read_text()
marker = "          python - <<'PY'\n"
s = t.index(marker) + len(marker)
tail = t[s:].splitlines()
end_idx = next(i for i, line in enumerate(tail) if line.strip() == 'PY')
code = textwrap.dedent('\n'.join(tail[:end_idx]))

extra = r'''
from pathlib import Path
import csv, math, statistics, collections

MASTER_PATH = Path('/tmp/MLB_Game_Stats_Joined_2021_2025.csv')
OUTDIR = Path('analysis_outputs/i2_market_pricing')
OUTDIR.mkdir(parents=True, exist_ok=True)

def truthy(v):
    return str(v).strip().lower() in {'1','true','t','yes','y'}

def fnum(v):
    try:
        if v is None or str(v).strip()=='':
            return None
        return float(v)
    except Exception:
        return None

def american_imp(o):
    if o is None:
        return None
    o=float(o)
    return (-o)/((-o)+100.0) if o < 0 else 100.0/(o+100.0)

def devig_under(over_odds, under_odds):
    po = american_imp(over_odds)
    pu = american_imp(under_odds)
    if po is None or pu is None or po+pu <= 0:
        return None
    return pu/(po+pu)

def quantile(vals, q):
    vals=sorted(vals)
    if not vals:
        return None
    if len(vals)==1:
        return vals[0]
    pos=(len(vals)-1)*q
    lo=int(math.floor(pos)); hi=int(math.ceil(pos))
    if lo==hi:
        return vals[lo]
    return vals[lo] + (vals[hi]-vals[lo])*(pos-lo)

def med(vals):
    vals=[v for v in vals if v is not None]
    return statistics.median(vals) if vals else None

def mean(vals):
    vals=[v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None

def pct(n,d):
    return 100.0*n/d if d else None

master={}
market_rows=[]
with MASTER_PATH.open(newline='',encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        gid=r.get('retro_game_id')
        if not gid:
            continue
        total=fnum(r.get('dk_total_open_total'))
        over=fnum(r.get('dk_total_open_overOdds'))
        undero=fnum(r.get('dk_total_open_underOdds'))
        i2=fnum(r.get('inning2_total_runs'))
        matched=truthy(r.get('benchmark_matched'))
        row={
            'gid':gid,
            'total':total,
            'over_odds':over,
            'under_odds':undero,
            'devig_under':devig_under(over,undero),
            'i2_under': (i2==0) if i2 is not None else None,
            'matched':matched,
        }
        master[gid]=row
        if matched and total is not None and row['i2_under'] is not None:
            market_rows.append(row)

by_total=collections.defaultdict(list)
by_total_devig=collections.defaultdict(list)
for r in market_rows:
    by_total[r['total']].append(1.0 if r['i2_under'] else 0.0)
    if r['devig_under'] is not None:
        by_total_devig[r['total']].append(r['devig_under'])
total_base={k:statistics.mean(v) for k,v in by_total.items()}
total_devig_mean={k:statistics.mean(v) for k,v in by_total_devig.items() if v}

num=0.0; den=0.0
for r in market_rows:
    x=r['devig_under']; t=r['total']
    if x is None or t not in total_base or t not in total_devig_mean:
        continue
    dx=x-total_devig_mean[t]
    dy=(1.0 if r['i2_under'] else 0.0)-total_base[t]
    num += dx*dy
    den += dx*dx
juice_beta = num/den if den else 0.0

def baseline_for_row(r, use_juice=False):
    if r is None or r['total'] not in total_base:
        return None
    b=total_base[r['total']]
    if use_juice and r['devig_under'] is not None and r['total'] in total_devig_mean:
        b += juice_beta*(r['devig_under']-total_devig_mean[r['total']])
    return max(0.01,min(0.99,b))

def summarize(rr):
    cohort_n=len(rr)
    cohort_under_n=sum(1 for g in rr if g['under'])
    cohort_under=cohort_under_n/cohort_n if cohort_n else None
    covered=[]
    for g in rr:
        m=master.get(g['gid'])
        if m and m['matched'] and m['total'] is not None and m['i2_under'] is not None:
            covered.append((g,m))
    covered_n=len(covered)
    covered_under=mean([1.0 if g['under'] else 0.0 for g,m in covered])
    totals=[m['total'] for g,m in covered]
    over_odds=[m['over_odds'] for g,m in covered if m['over_odds'] is not None]
    under_odds=[m['under_odds'] for g,m in covered if m['under_odds'] is not None]
    dv=[m['devig_under'] for g,m in covered if m['devig_under'] is not None]
    base_total=[baseline_for_row(m,False) for g,m in covered]
    base_juice=[baseline_for_row(m,True) for g,m in covered]
    base_total_mean=mean(base_total)
    base_juice_mean=mean(base_juice)
    return {
        'cohort_n':cohort_n,
        'market_n':covered_n,
        'cohort_under_pct':100*cohort_under if cohort_under is not None else None,
        'market_covered_under_pct':100*covered_under if covered_under is not None else None,
        'fair_under': fair(cohort_under) if cohort_under is not None else None,
        'median_total':med(totals),
        'q1_total':quantile(totals,.25),
        'q3_total':quantile(totals,.75),
        'median_over_odds':med(over_odds),
        'median_under_odds':med(under_odds),
        'median_devig_under_pct':100*med(dv) if dv else None,
        'baseline_total_pct':100*base_total_mean if base_total_mean is not None else None,
        'baseline_total_juice_pct':100*base_juice_mean if base_juice_mean is not None else None,
        'residual_all_vs_total_pp':100*(cohort_under-base_total_mean) if cohort_under is not None and base_total_mean is not None else None,
        'residual_covered_vs_total_pp':100*(covered_under-base_total_mean) if covered_under is not None and base_total_mean is not None else None,
        'residual_all_vs_juice_pp':100*(cohort_under-base_juice_mean) if cohort_under is not None and base_juice_mean is not None else None,
        'residual_covered_vs_juice_pp':100*(covered_under-base_juice_mean) if covered_under is not None and base_juice_mean is not None else None,
        'share_le_7_5_pct':pct(sum(1 for x in totals if x<=7.5),len(totals)),
        'share_eq_8_0_pct':pct(sum(1 for x in totals if x==8.0),len(totals)),
        'share_eq_8_5_pct':pct(sum(1 for x in totals if x==8.5),len(totals)),
        'share_ge_9_0_pct':pct(sum(1 for x in totals if x>=9.0),len(totals)),
    }

expected = {
    ('both_top3','0-1'):(90,64),
    ('both_top3','2+'):(78,50),
    ('any_back','0-1'):(122,76),
    ('any_back','2+'):(113,66),
}
expected_sun = {
    ('both_top3','0-1'):(44,33),
    ('both_top3','2+'):(37,24),
    ('any_back','0-1'):(67,43),
    ('any_back','2+'):(56,26),
}
for key,(n,u) in expected.items():
    tl,rb=key
    rr=[g for g in valid if g.get('sg')==4 and
        ((g['both_top3']) if tl=='both_top3' else (g['any_back'])) and
        g.get('rest_total') is not None and
        ((g['rest_total']<=1) if rb=='0-1' else (g['rest_total']>=2))]
    assert len(rr)==n and sum(1 for g in rr if g['under'])==u, (key,len(rr),sum(1 for g in rr if g['under']))
for key,(n,u) in expected_sun.items():
    tl,rb=key
    rr=[g for g in valid if g.get('sg')==4 and g.get('day')=='Sunday' and
        ((g['both_top3']) if tl=='both_top3' else (g['any_back'])) and
        g.get('rest_total') is not None and
        ((g['rest_total']<=1) if rb=='0-1' else (g['rest_total']>=2))]
    assert len(rr)==n and sum(1 for g in rr if g['under'])==u, ('Sunday',key,len(rr),sum(1 for g in rr if g['under']))

fields=[
    'series_game','rotation_tier','rest_bucket','cohort_n','market_n',
    'cohort_under_pct','market_covered_under_pct','fair_under',
    'median_total','q1_total','q3_total',
    'median_over_odds','median_under_odds','median_devig_under_pct',
    'baseline_total_pct','baseline_total_juice_pct',
    'residual_all_vs_total_pp','residual_covered_vs_total_pp',
    'residual_all_vs_juice_pp','residual_covered_vs_juice_pp',
    'share_le_7_5_pct','share_eq_8_0_pct','share_eq_8_5_pct','share_ge_9_0_pct'
]
rows=[]
for sg in [1,2,3,4]:
    for tl in ['both_top3','any_back']:
        for rb in ['0-1','2+']:
            rr=[g for g in valid if g.get('sg')==sg and
                ((g['both_top3']) if tl=='both_top3' else (g['any_back'])) and
                g.get('rest_total') is not None and
                ((g['rest_total']<=1) if rb=='0-1' else (g['rest_total']>=2))]
            s=summarize(rr); s.update({'series_game':sg,'rotation_tier':tl,'rest_bucket':rb})
            rows.append(s)

with (OUTDIR/'sixteen_cell_market_pricing.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for r in rows: w.writerow({k:r.get(k) for k in fields})

cmp_fields=[
    'rotation_tier','rest_bucket',
    'g1_3_n','g4_n','g1_3_under_pct','g4_under_pct','under_delta_pp',
    'g1_3_market_n','g4_market_n','g1_3_median_total','g4_median_total','median_total_delta',
    'g1_3_baseline_total_pct','g4_baseline_total_pct','baseline_delta_pp',
    'g1_3_residual_total_pp','g4_residual_total_pp','residual_delta_pp',
    'g1_3_baseline_juice_pct','g4_baseline_juice_pct',
    'g1_3_residual_juice_pp','g4_residual_juice_pp','residual_juice_delta_pp'
]
cmp=[]
for tl in ['both_top3','any_back']:
    for rb in ['0-1','2+']:
        def pred(g):
            return ((g['both_top3']) if tl=='both_top3' else (g['any_back'])) and g.get('rest_total') is not None and (((g['rest_total']<=1) if rb=='0-1' else (g['rest_total']>=2)))
        a=[g for g in valid if g.get('sg') in [1,2,3] and pred(g)]
        b=[g for g in valid if g.get('sg')==4 and pred(g)]
        sa=summarize(a); sb=summarize(b)
        x={
            'rotation_tier':tl,'rest_bucket':rb,
            'g1_3_n':sa['cohort_n'],'g4_n':sb['cohort_n'],
            'g1_3_under_pct':sa['cohort_under_pct'],'g4_under_pct':sb['cohort_under_pct'],
            'under_delta_pp':sb['cohort_under_pct']-sa['cohort_under_pct'],
            'g1_3_market_n':sa['market_n'],'g4_market_n':sb['market_n'],
            'g1_3_median_total':sa['median_total'],'g4_median_total':sb['median_total'],
            'median_total_delta':(sb['median_total']-sa['median_total']) if sa['median_total'] is not None and sb['median_total'] is not None else None,
            'g1_3_baseline_total_pct':sa['baseline_total_pct'],'g4_baseline_total_pct':sb['baseline_total_pct'],
            'baseline_delta_pp':sb['baseline_total_pct']-sa['baseline_total_pct'],
            'g1_3_residual_total_pp':sa['residual_all_vs_total_pp'],'g4_residual_total_pp':sb['residual_all_vs_total_pp'],
            'residual_delta_pp':sb['residual_all_vs_total_pp']-sa['residual_all_vs_total_pp'],
            'g1_3_baseline_juice_pct':sa['baseline_total_juice_pct'],'g4_baseline_juice_pct':sb['baseline_total_juice_pct'],
            'g1_3_residual_juice_pp':sa['residual_all_vs_juice_pp'],'g4_residual_juice_pp':sb['residual_all_vs_juice_pp'],
            'residual_juice_delta_pp':sb['residual_all_vs_juice_pp']-sa['residual_all_vs_juice_pp'],
        }
        cmp.append(x)
with (OUTDIR/'game4_vs_g1_3.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=cmp_fields); w.writeheader(); w.writerows(cmp)

sun_rows=[]
for tl in ['both_top3','any_back']:
    for rb in ['0-1','2+']:
        rr=[g for g in valid if g.get('sg')==4 and g.get('day')=='Sunday' and
            ((g['both_top3']) if tl=='both_top3' else (g['any_back'])) and
            g.get('rest_total') is not None and
            ((g['rest_total']<=1) if rb=='0-1' else (g['rest_total']>=2))]
        s=summarize(rr); s.update({'series_game':4,'rotation_tier':tl,'rest_bucket':rb})
        sun_rows.append(s)
with (OUTDIR/'sunday_game4_market_pricing.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for r in sun_rows: w.writerow({k:r.get(k) for k in fields})

with (OUTDIR/'diagnostics.txt').open('w',encoding='utf-8') as f:
    f.write(f"market_rows={len(market_rows)}\n")
    f.write(f"juice_beta={juice_beta}\n")
    for tt in sorted(total_base):
        if 6.0 <= tt <= 11.0:
            f.write(f"total={tt:.1f},n={len(by_total[tt])},i2_under_pct={100*total_base[tt]:.8f},mean_devig_under_pct={100*total_devig_mean.get(tt,float('nan')):.8f}\n")

def fmt(v, nd=2):
    if v is None:
        return ''
    if isinstance(v,int):
        return str(v)
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)

md=[]
md.append("# I2 Market-Pricing Test")
md.append("")
md.append(f"Market-covered master rows: {len(market_rows)}")
md.append(f"Within-total de-vig juice adjustment slope: {juice_beta:.6f}")
md.append("")
md.append("## 16-cell table")
md.append("")
show=['series_game','rotation_tier','rest_bucket','cohort_n','market_n','cohort_under_pct','median_total','median_devig_under_pct','baseline_total_pct','baseline_total_juice_pct','residual_all_vs_total_pp','residual_all_vs_juice_pp','fair_under']
md.append("|"+"|".join(show)+"|")
md.append("|"+"|".join(["---"]*len(show))+"|")
for r in rows:
    md.append("|"+"|".join(fmt(r.get(k),2) for k in show)+"|")
md.append("")
md.append("## Game 4 vs pooled G1-G3")
md.append("")
show2=['rotation_tier','rest_bucket','g1_3_under_pct','g4_under_pct','under_delta_pp','g1_3_median_total','g4_median_total','median_total_delta','baseline_delta_pp','g1_3_residual_total_pp','g4_residual_total_pp','residual_delta_pp','g1_3_residual_juice_pp','g4_residual_juice_pp','residual_juice_delta_pp']
md.append("|"+"|".join(show2)+"|")
md.append("|"+"|".join(["---"]*len(show2))+"|")
for r in cmp:
    md.append("|"+"|".join(fmt(r.get(k),2) for k in show2)+"|")
md.append("")
md.append("## Sunday Game 4")
md.append("")
md.append("|"+"|".join(show)+"|")
md.append("|"+"|".join(["---"]*len(show))+"|")
for r in sun_rows:
    md.append("|"+"|".join(fmt(r.get(k),2) for k in show)+"|")
md.append("")
(OUTDIR/'REPORT.md').write_text("\n".join(md)+"\n",encoding='utf-8')
print("\nI2_MARKET_PRICING_OUTPUT", OUTDIR)
'''

exec(compile(code + '\n' + extra, '<i2-market-pricing>', 'exec'), globals(), globals())
