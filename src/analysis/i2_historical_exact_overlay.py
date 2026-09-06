#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

BUCKETS = ["0","1","2","3","4+"]

def game_key_col(df):
    for c in ["game_id","gamePk","game_pk","gamepk","pk"]:
        if c in df.columns:
            return c
    return None

def find_outcomes(root: Path):
    diagnostics=[]
    for p in root.rglob("*.csv"):
        try:
            df=pd.read_csv(p, nrows=10)
        except Exception:
            continue
        cols=list(df.columns)
        key=game_key_col(df)
        diagnostics.append((str(p), cols))
        if key and "inning2_total_runs" in cols:
            return p, "direct", key
        if key and "i2_runs" in cols:
            return p, "direct_i2", key
        if key and "inning" in cols and any(c in cols for c in ["runs","inning_runs","total_runs"]):
            return p, "long", key
    # JSON fallback
    for p in root.rglob("*.json"):
        try:
            obj=json.loads(p.read_text())
        except Exception:
            continue
        rows=obj if isinstance(obj,list) else None
        if rows and isinstance(rows[0],dict):
            cols=list(rows[0].keys()); key=None
            for c in ["game_id","gamePk","game_pk","gamepk","pk"]:
                if c in cols: key=c; break
            if key and ("inning2_total_runs" in cols or "i2_runs" in cols):
                return p, "json_direct", key
    print("Could not identify numeric game-keyed inning outcome file. CSV diagnostics:")
    for path,cols in diagnostics:
        if any("game" in c.lower() for c in cols) or any("inning" in c.lower() for c in cols):
            print(path, cols[:40])
    raise SystemExit(2)

def load_outcomes(path, mode, key):
    if mode.startswith("json"):
        obj=json.loads(Path(path).read_text())
        df=pd.DataFrame(obj if isinstance(obj,list) else obj.get("rows",[]))
    else:
        df=pd.read_csv(path)
    if mode=="direct":
        out=df[[key,"season","inning2_total_runs"]].copy() if "season" in df.columns else df[[key,"inning2_total_runs"]].copy()
        out=out.rename(columns={key:"game_id","inning2_total_runs":"i2_runs"})
    elif mode in ["direct_i2","json_direct"]:
        keep=[key,"i2_runs"] + (["season"] if "season" in df.columns else [])
        out=df[keep].copy().rename(columns={key:"game_id"})
    elif mode=="long":
        runcol=next(c for c in ["runs","inning_runs","total_runs"] if c in df.columns)
        d=df.copy()
        # accept inning as 2 / I2 / second
        mask=d["inning"].astype(str).str.lower().isin(["2","i2","second","2.0"])
        d=d[mask]
        groupcols=[key] + (["season"] if "season" in d.columns else [])
        out=d.groupby(groupcols,as_index=False)[runcol].sum().rename(columns={key:"game_id",runcol:"i2_runs"})
    else:
        raise ValueError(mode)
    out["game_id"]=pd.to_numeric(out["game_id"],errors="coerce")
    out["i2_runs"]=pd.to_numeric(out["i2_runs"],errors="coerce")
    return out.dropna(subset=["game_id","i2_runs"]).drop_duplicates("game_id")

def prior_tail_shares(history, season):
    h=history[(history["season"]<season) & (history["i2_runs"]>0)].copy()
    if h.empty:
        raise ValueError(f"No prior-season positive-run history for {season}")
    cats=pd.Series(np.where(h.i2_runs>=4,"4+",h.i2_runs.astype(int).astype(str)))
    vc=cats.value_counts()
    arr=np.array([vc.get("1",0),vc.get("2",0),vc.get("3",0),vc.get("4+",0)],dtype=float)
    return arr/arr.sum()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--phase1",required=True)
    ap.add_argument("--validation",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    phase1=Path(args.phase1); validation=Path(args.validation); outdir=Path(args.out); outdir.mkdir(parents=True,exist_ok=True)
    predpath=next(validation.rglob("v04_oos_predictions.csv"))
    pred=pd.read_csv(predpath)
    pred["game_id"]=pd.to_numeric(pred["game_id"],errors="coerce")
    pred["season"]=pd.to_numeric(pred["season"],errors="coerce").astype(int)
    opath,mode,key=find_outcomes(phase1)
    hist=load_outcomes(opath,mode,key)
    if "season" not in hist.columns:
        raise SystemExit(f"Outcome source {opath} lacks season; cannot perform leakage-safe prior-tail allocation")
    hist["season"]=pd.to_numeric(hist["season"],errors="coerce").astype(int)
    m=pred.merge(hist[["game_id","season","i2_runs"]],on=["game_id","season"],how="left",validate="one_to_one")
    joinrate=m.i2_runs.notna().mean()
    print(f"Outcome source: {opath} mode={mode}; join rate={joinrate:.6f} ({m.i2_runs.notna().sum()}/{len(m)})")
    if joinrate<0.99:
        print(m[m.i2_runs.isna()].head(20).to_string(index=False))
        raise SystemExit(3)
    m["actual_bucket"]=np.where(m.i2_runs>=4,"4+",m.i2_runs.astype(int).astype(str))
    expected={b:0.0 for b in BUCKETS}
    tail_by_season={}
    for season,g in m.groupby("season"):
        shares=prior_tail_shares(hist,season)
        tail_by_season[str(season)]={b:float(v) for b,v in zip(["1","2","3","4+"],shares)}
        p0=g.p_under_local_cv.to_numpy(float)
        expected["0"] += p0.sum()
        pover=1-p0
        for b,s in zip(["1","2","3","4+"],shares): expected[b] += (pover*s).sum()
    actual=m.actual_bucket.value_counts()
    rows=[]
    N=len(m)
    for b in BUCKETS:
        a=int(actual.get(b,0)); e=float(expected[b])
        rows.append({"i2_runs":b,"actual_games":a,"predicted_expected_games":e,"actual_pct":100*a/N,"predicted_pct":100*e/N,"actual_minus_predicted_games":a-e,"actual_minus_predicted_pp":100*(a-e)/N})
    res=pd.DataFrame(rows)
    res.to_csv(outdir/"historical_i2_actual_vs_predicted_exact.csv",index=False)
    m[["game_id","season","p_under_local_cv","i2_runs","actual_bucket"]].to_csv(outdir/"historical_i2_overlay_game_level.csv",index=False)
    manifest={
      "model":"I2 v0.4 Local-CV",
      "n":N,
      "seasons":sorted(map(int,m.season.unique())),
      "outcome_source":str(opath),
      "outcome_join_rate":joinrate,
      "prediction_zero_bucket":"exact v0.4 Local-CV P(Under0.5)",
      "positive_tail_method":"For each OOS season, allocate that game's v0.4 P(Over0.5) proportionally across 1/2/3/4+ using the empirical positive-run distribution from prior seasons only. This mirrors the production model's proportional positive-tail conditioning principle but is not a byte-for-byte historical replay of the live lineup simulator exact-run output.",
      "tail_shares_by_season":tail_by_season,
      "price_used":False
    }
    (outdir/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(res.to_string(index=False))
    print(json.dumps(manifest,indent=2))

if __name__=="__main__": main()
