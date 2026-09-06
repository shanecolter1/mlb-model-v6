#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

BUCKETS = ["0","1","2","3","4+"]

def game_key_col(df):
    for c in ["game_id","gamePk","game_pk","gamepk","pk","game_id_numeric"]:
        if c in df.columns:
            return c
    return None

def classify(df):
    cols=list(df.columns); key=game_key_col(df)
    if key and "inning2_total_runs" in cols: return "direct", key
    if key and "i2_runs" in cols: return "direct_i2", key
    if key and "inning" in cols and any(c in cols for c in ["runs","inning_runs","total_runs","runs_total"]): return "long", key
    # Common Phase-1 inning result schema: game key + inning number + away/home runs.
    if key and "inning" in cols and any(c in cols for c in ["away_runs","awayRuns"]) and any(c in cols for c in ["home_runs","homeRuns"]): return "long_sides", key
    return None, key

def read_any(p, small=False):
    if p.suffix==".parquet":
        return pd.read_parquet(p)
    if p.name.endswith(".csv.gz") or p.suffix==".csv":
        return pd.read_csv(p, nrows=20 if small else None)
    raise ValueError(p)

def find_outcomes(root: Path):
    diagnostics=[]
    candidates=list(root.rglob("*.parquet"))+list(root.rglob("*.csv"))+list(root.rglob("*.csv.gz"))
    # Prefer explicit inning/results files.
    candidates.sort(key=lambda p: (0 if ("inning" in p.name.lower() or "result" in p.name.lower()) else 1, str(p)))
    for p in candidates:
        try: df=read_any(p, small=True)
        except Exception as e:
            diagnostics.append((str(p), [f"READ_ERROR:{e}"]))
            continue
        cols=list(df.columns); diagnostics.append((str(p), cols))
        mode,key=classify(df)
        if mode: return p,mode,key
    print("Could not identify numeric game-keyed inning outcome file. Diagnostics:")
    for path,cols in diagnostics: print(path, cols[:60])
    raise SystemExit(2)

def load_outcomes(path, mode, key):
    df=read_any(Path(path), small=False)
    if mode=="direct":
        keep=[key,"inning2_total_runs"] + (["season"] if "season" in df.columns else [])
        out=df[keep].copy().rename(columns={key:"game_id","inning2_total_runs":"i2_runs"})
    elif mode=="direct_i2":
        keep=[key,"i2_runs"] + (["season"] if "season" in df.columns else [])
        out=df[keep].copy().rename(columns={key:"game_id"})
    elif mode=="long":
        runcol=next(c for c in ["runs","inning_runs","total_runs","runs_total"] if c in df.columns)
        d=df[df["inning"].astype(str).str.lower().isin(["2","i2","second","2.0"])].copy()
        groupcols=[key] + (["season"] if "season" in d.columns else [])
        # If one row per half, sum; if one row per full inning, group still works.
        out=d.groupby(groupcols,as_index=False)[runcol].sum().rename(columns={key:"game_id",runcol:"i2_runs"})
    elif mode=="long_sides":
        away=next(c for c in ["away_runs","awayRuns"] if c in df.columns)
        home=next(c for c in ["home_runs","homeRuns"] if c in df.columns)
        d=df[df["inning"].astype(str).str.lower().isin(["2","i2","second","2.0"])].copy()
        d["i2_runs"]=pd.to_numeric(d[away],errors="coerce").fillna(0)+pd.to_numeric(d[home],errors="coerce").fillna(0)
        keep=[key,"i2_runs"] + (["season"] if "season" in d.columns else [])
        out=d[keep].copy().rename(columns={key:"game_id"}).drop_duplicates("game_id")
    else: raise ValueError(mode)
    out["game_id"]=pd.to_numeric(out["game_id"],errors="coerce")
    out["i2_runs"]=pd.to_numeric(out["i2_runs"],errors="coerce")
    return out.dropna(subset=["game_id","i2_runs"]).drop_duplicates("game_id")

def attach_season(hist, root):
    if "season" in hist.columns: return hist
    for p in root.rglob("games.parquet"):
        g=pd.read_parquet(p); key=game_key_col(g)
        if not key: continue
        season_col=next((c for c in ["season","year"] if c in g.columns),None)
        date_col=next((c for c in ["game_date","date","officialDate"] if c in g.columns),None)
        gg=g[[key]+([season_col] if season_col else [date_col])].copy().rename(columns={key:"game_id"})
        gg["game_id"]=pd.to_numeric(gg["game_id"],errors="coerce")
        if season_col: gg["season"]=pd.to_numeric(gg[season_col],errors="coerce")
        else: gg["season"]=pd.to_datetime(gg[date_col],errors="coerce").dt.year
        return hist.merge(gg[["game_id","season"]].drop_duplicates("game_id"),on="game_id",how="left")
    return hist

def prior_tail_shares(history, season):
    h=history[(history["season"]<season)&(history["i2_runs"]>0)].copy()
    if h.empty: raise ValueError(f"No prior-season positive-run history for {season}")
    cats=pd.Series(np.where(h.i2_runs>=4,"4+",h.i2_runs.astype(int).astype(str)))
    vc=cats.value_counts(); arr=np.array([vc.get("1",0),vc.get("2",0),vc.get("3",0),vc.get("4+",0)],float)
    return arr/arr.sum()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--phase1",required=True); ap.add_argument("--validation",required=True); ap.add_argument("--out",required=True); args=ap.parse_args()
    phase1=Path(args.phase1); validation=Path(args.validation); outdir=Path(args.out); outdir.mkdir(parents=True,exist_ok=True)
    pred=pd.read_csv(next(validation.rglob("v04_oos_predictions.csv")))
    pred["game_id"]=pd.to_numeric(pred["game_id"],errors="coerce"); pred["season"]=pd.to_numeric(pred["season"],errors="coerce").astype(int)
    opath,mode,key=find_outcomes(phase1); hist=attach_season(load_outcomes(opath,mode,key),phase1)
    if "season" not in hist.columns or hist["season"].isna().all(): raise SystemExit(f"Outcome source {opath} lacks resolvable season")
    hist["season"]=pd.to_numeric(hist["season"],errors="coerce").astype("Int64")
    m=pred.merge(hist[["game_id","season","i2_runs"]].dropna().astype({"season":int}),on=["game_id","season"],how="left",validate="one_to_one")
    joinrate=m.i2_runs.notna().mean(); print(f"Outcome source: {opath} mode={mode}; join rate={joinrate:.6f} ({m.i2_runs.notna().sum()}/{len(m)})")
    if joinrate<0.99:
        print(m[m.i2_runs.isna()].head(20).to_string(index=False)); raise SystemExit(3)
    m["actual_bucket"]=np.where(m.i2_runs>=4,"4+",m.i2_runs.astype(int).astype(str))
    expected={b:0.0 for b in BUCKETS}; tail_by_season={}
    for season,g in m.groupby("season"):
        shares=prior_tail_shares(hist.astype({"season":int}),season); tail_by_season[str(season)]={b:float(v) for b,v in zip(["1","2","3","4+"],shares)}
        p0=g.p_under_local_cv.to_numpy(float); expected["0"]+=p0.sum(); pover=1-p0
        for b,s in zip(["1","2","3","4+"],shares): expected[b]+=(pover*s).sum()
    actual=m.actual_bucket.value_counts(); N=len(m); rows=[]
    for b in BUCKETS:
        a=int(actual.get(b,0)); e=float(expected[b]); rows.append({"i2_runs":b,"actual_games":a,"predicted_expected_games":e,"actual_pct":100*a/N,"predicted_pct":100*e/N,"actual_minus_predicted_games":a-e,"actual_minus_predicted_pp":100*(a-e)/N})
    res=pd.DataFrame(rows); res.to_csv(outdir/"historical_i2_actual_vs_predicted_exact.csv",index=False)
    m[["game_id","season","p_under_local_cv","i2_runs","actual_bucket"]].to_csv(outdir/"historical_i2_overlay_game_level.csv",index=False)
    manifest={"model":"I2 v0.4 Local-CV","n":N,"seasons":sorted(map(int,m.season.unique())),"outcome_source":str(opath),"outcome_join_rate":joinrate,"prediction_zero_bucket":"exact v0.4 Local-CV P(Under0.5)","positive_tail_method":"For each OOS season, allocate each game's v0.4 P(Over0.5) across 1/2/3/4+ using the empirical positive-run distribution from prior seasons only. This mirrors the production proportional positive-tail conditioning principle; it is a leakage-safe reconstruction, not a byte-for-byte historical replay of live-simulator exact-run output.","tail_shares_by_season":tail_by_season,"price_used":False}
    (outdir/"manifest.json").write_text(json.dumps(manifest,indent=2)); print(res.to_string(index=False)); print(json.dumps(manifest,indent=2))

if __name__=="__main__": main()
