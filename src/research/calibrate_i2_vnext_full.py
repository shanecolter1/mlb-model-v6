#!/usr/bin/env python3
"""Estimate the single final full-I2 calibration curve for I2 vNext.

Uses 2025 normal-starter games only. The PA event model is trained on 2023-24,
then 2025 full-I2 raw probabilities are replayed from Retrosheet pregame
lineups and observed I2 starting slots. The I2 outcome is never used as a
feature. Calibration is fit on the first chronological half of 2025 and
validated on the second half.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/research"))
from fit_i2_vnext import add_arsenal_feature, arsenal_maps, fit_one, matchup_score, score

EVENTS = [
    "single", "double", "triple", "home_run", "walk", "hit_by_pitch",
    "strikeout", "ball_in_play_out",
]
CHADWICK_BASE = "https://raw.githubusercontent.com/chadwickbureau/register/master/data"
RETRO_URL = "https://www.retrosheet.org/downloads/plays/2025plays.zip"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("data/derived/i2_vnext/i2_pa_statcast.csv"))
    p.add_argument("--arsenal-dir", type=Path, default=Path("data/derived/i2_vnext/arsenal"))
    p.add_argument("--play-calibration", type=Path, default=Path("data/derived/i2/i2_play_calibration.json"))
    p.add_argument("--output", type=Path, default=Path("data/derived/i2_vnext/full_model_calibration.json"))
    p.add_argument("--half-lives", default="180,365,730,1460")
    p.add_argument("--c-grid", default="0.05,0.2,1.0")
    return p.parse_args()


def download_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=120, headers={"User-Agent": "MLB-I2-vNext/1.0"})
    r.raise_for_status()
    return r.content


def chadwick_map() -> dict[str, int]:
    out = {}
    for suffix in "0123456789abcdef":
        url = f"{CHADWICK_BASE}/people-{suffix}.csv"
        r = requests.get(url, timeout=60, headers={"User-Agent": "MLB-I2-vNext/1.0"})
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), low_memory=False, usecols=lambda c: c in {"key_retro", "key_mlbam"})
        df = df.dropna(subset=["key_retro", "key_mlbam"])
        for rr in df.itertuples(index=False):
            try:
                out[str(rr.key_retro)] = int(rr.key_mlbam)
            except Exception:
                pass
    return out


def as_int(v):
    try:
        return int(v or 0)
    except Exception:
        return 0


def retrosheet_cases(blob: bytes, idmap: dict[str, int]) -> list[dict]:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        with zf.open("2025plays.csv") as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            by_side = defaultdict(lambda: {1: [], 2: []})
            for row in reader:
                if row.get("gametype") != "regular":
                    continue
                inning = as_int(row.get("inning"))
                if inning not in (1, 2):
                    continue
                by_side[(row["gid"], as_int(row.get("top_bot")))][inning].append(row)

    halves = {}
    for (gid, top_bot), innings in by_side.items():
        i1, i2 = innings[1], innings[2]
        if not i1 or not i2:
            continue
        first1, first2 = i1[0], i2[0]
        retro_lineup = [first1.get(f"l{k}") for k in range(1, 10)]
        if any(not x for x in retro_lineup):
            continue
        lineup = [idmap.get(x) for x in retro_lineup]
        i2_pitcher = idmap.get(first2.get("pitcher"))
        i1_pitcher = idmap.get(first1.get("pitcher"))
        if any(x is None for x in lineup) or i2_pitcher is None or i1_pitcher is None:
            continue
        if i1_pitcher != i2_pitcher:
            continue
        slot_map = {b: idx + 1 for idx, b in enumerate(retro_lineup)}
        start_slot = slot_map.get(first2.get("batter")) or as_int(first2.get("lp"))
        if not 1 <= int(start_slot) <= 9:
            continue
        halves[(gid, top_bot)] = {
            "gid": gid,
            "date": str(first1.get("date")),
            "lineup": [int(x) for x in lineup],
            "pitcher": int(i2_pitcher),
            "start_slot": int(start_slot),
            "runs": sum(as_int(r.get("runs")) for r in i2),
        }

    games = []
    for gid in sorted({g for g, _ in halves}):
        top, bottom = halves.get((gid, 0)), halves.get((gid, 1))
        if top and bottom:
            games.append({"gid": gid, "date": top["date"], "top": top, "bottom": bottom})
    return games


def player_sides(df: pd.DataFrame):
    pitcher_throws = {}
    for pid, g in df.groupby("pitcher"):
        vals = [str(x) for x in g["p_throws"].dropna() if str(x) in {"L", "R"}]
        if vals:
            pitcher_throws[int(pid)] = Counter(vals).most_common(1)[0][0]
    batter_bats = {}
    for bid, g in df.groupby("batter"):
        vals = {str(x) for x in g["stand"].dropna() if str(x) in {"L", "R"}}
        if vals == {"L", "R"}:
            batter_bats[int(bid)] = "S"
        elif vals:
            batter_bats[int(bid)] = next(iter(vals))
    return batter_bats, pitcher_throws


def batter_side(batter: int, throws: str, bats: dict[int, str]) -> str:
    side = bats.get(batter, "R")
    if side == "S":
        return "L" if throws == "R" else "R"
    return side


def prepare_train(df: pd.DataFrame, arsenal_dir: Path):
    train = df[df["season"].isin([2023, 2024])].copy()
    train["batter"] = pd.to_numeric(train["batter"], errors="raise").astype(int)
    train["pitcher"] = pd.to_numeric(train["pitcher"], errors="raise").astype(int)
    train = add_arsenal_feature(train, arsenal_dir)
    mean = float(train["arsenal_matchup_xwoba"].mean())
    sd = float(train["arsenal_matchup_xwoba"].std(ddof=0)) or 1.0
    train["arsenal_z"] = (train["arsenal_matchup_xwoba"] - mean) / sd
    train["batter"] = train["batter"].astype(str)
    train["pitcher"] = train["pitcher"].astype(str)
    train["platoon"] = train["platoon"].fillna("?v?").astype(str)
    return train, mean, sd


def select_hyperparams(train: pd.DataFrame, half_lives: list[float], cs: list[float]):
    tr = train[train["season"] == 2023]
    te = train[train["season"] == 2024]
    if tr.empty or te.empty:
        raise RuntimeError("Need both 2023 and 2024 for chronological hyperparameter selection")
    best = None
    table = []
    for half_life in half_lives:
        for c in cs:
            prep, model = fit_one(tr, c, half_life)
            result = score(prep, model, te)
            row = {"half_life_days": half_life, "C": c, **result}
            table.append(row)
            key = (result["logloss"], result["brier_multiclass"], half_life, c)
            if best is None or key < best[0]:
                best = (key, half_life, c)
    return best[1], best[2], table


def load_transitions(path: Path):
    return json.loads(path.read_text())["base_transitions"]


def event_vector(prep, model, batter, pitcher, side, throws, arsenal, mean, sd):
    score_x = matchup_score(batter, pitcher, arsenal)
    row = pd.DataFrame([{
        "batter": str(batter),
        "pitcher": str(pitcher),
        "platoon": f"{side}v{throws}",
        "arsenal_z": (score_x - mean) / sd,
    }])
    p = model.predict_proba(prep.transform(row[["batter", "pitcher", "platoon", "arsenal_z"]]))[0]
    return {str(cls): float(prob) for cls, prob in zip(model.classes_, p)}


def p_scoreless_half(case, prep, model, arsenal, mean, sd, bats, throws_map, transitions):
    pitcher = case["pitcher"]
    throws = throws_map.get(pitcher, "R")
    cache = {}
    states = {(0, 0, case["start_slot"]): 1.0}
    finished = 0.0
    for _ in range(40):
        if not states:
            break
        nxt = defaultdict(float)
        for (outs, mask, slot), mass in states.items():
            batter = case["lineup"][slot - 1]
            side = batter_side(batter, throws, bats)
            key = (batter, pitcher, side, throws)
            vec = cache.get(key)
            if vec is None:
                vec = event_vector(prep, model, batter, pitcher, side, throws, arsenal, mean, sd)
                cache[key] = vec
            next_slot = 1 if slot == 9 else slot + 1
            for event, event_p in vec.items():
                for opt in transitions.get(f"{event}|{outs}|{mask}") or []:
                    p = mass * event_p * float(opt.get("p", 0))
                    if p <= 0 or int(opt.get("runs", 0)) > 0:
                        continue
                    new_outs = min(3, outs + int(opt.get("outs_added", 0)))
                    if new_outs >= 3:
                        finished += p
                    else:
                        nxt[(new_outs, int(opt.get("post_mask", 0)), next_slot)] += p
        states = dict(nxt)
        if sum(states.values()) < 1e-12:
            break
    return max(0.0, min(1.0, finished))


def clip_prob(p):
    return max(1e-6, min(1 - 1e-6, float(p)))


def logits(values):
    p = np.asarray([clip_prob(x) for x in values], dtype=float)
    return np.log(p / (1 - p)).reshape(-1, 1)


def metrics(y, p):
    p = np.asarray([clip_prob(x) for x in p], dtype=float)
    y = np.asarray(y, dtype=int)
    return {
        "n": int(len(y)),
        "brier": float(np.mean((p - y) ** 2)),
        "logloss": float(log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1])),
        "predicted_mean": float(np.mean(p)),
        "realized_rate": float(np.mean(y)),
    }


def main():
    args = parse_args()
    df = pd.read_csv(args.dataset)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    bats, pitcher_throws = player_sides(df)

    train, mean, sd = prepare_train(df, args.arsenal_dir)
    half_life, c, grid = select_hyperparams(
        train,
        [float(x) for x in args.half_lives.split(",") if x],
        [float(x) for x in args.c_grid.split(",") if x],
    )
    prep, model = fit_one(train, c, half_life)
    arsenal_2024 = arsenal_maps(
        args.arsenal_dir / "batter_2024.csv",
        args.arsenal_dir / "pitcher_2024.csv",
    )
    transitions = load_transitions(args.play_calibration)
    games = retrosheet_cases(download_bytes(RETRO_URL), chadwick_map())

    rows = []
    for game in games:
        top_p0 = p_scoreless_half(
            game["top"], prep, model, arsenal_2024, mean, sd,
            bats, pitcher_throws, transitions,
        )
        bottom_p0 = p_scoreless_half(
            game["bottom"], prep, model, arsenal_2024, mean, sd,
            bats, pitcher_throws, transitions,
        )
        rows.append({
            "gid": game["gid"],
            "date": game["date"],
            "raw_under": top_p0 * bottom_p0,
            "realized_under": int(game["top"]["runs"] == 0 and game["bottom"]["runs"] == 0),
        })

    replay = pd.DataFrame(rows).sort_values(["date", "gid"]).reset_index(drop=True)
    if len(replay) < 500:
        raise RuntimeError(f"Insufficient 2025 replay sample: {len(replay)}")

    cut = len(replay) // 2
    cal, val = replay.iloc[:cut], replay.iloc[cut:]
    y_cal = cal["realized_under"].to_numpy(int)
    y_val = val["realized_under"].to_numpy(int)
    p_cal = cal["raw_under"].to_numpy(float)
    p_val = val["raw_under"].to_numpy(float)

    candidates = {"none": {"validation": metrics(y_val, p_val)}}

    sigmoid = LogisticRegression(C=1e6, solver="lbfgs").fit(logits(p_cal), y_cal)
    p_sigmoid = sigmoid.predict_proba(logits(p_val))[:, 1]
    candidates["sigmoid"] = {
        "validation": metrics(y_val, p_sigmoid),
        "intercept": float(sigmoid.intercept_[0]),
        "slope": float(sigmoid.coef_[0, 0]),
    }

    isotonic = IsotonicRegression(out_of_bounds="clip").fit(p_cal, y_cal)
    p_isotonic = isotonic.predict(p_val)
    candidates["isotonic"] = {
        "validation": metrics(y_val, p_isotonic),
        "x_thresholds": [float(x) for x in isotonic.X_thresholds_],
        "y_thresholds": [float(x) for x in isotonic.y_thresholds_],
    }

    chosen = min(
        candidates,
        key=lambda k: (
            candidates[k]["validation"]["logloss"],
            candidates[k]["validation"]["brier"],
        ),
    )

    final = {"type": chosen}
    if chosen == "sigmoid":
        m = LogisticRegression(C=1e6, solver="lbfgs").fit(
            logits(replay["raw_under"]),
            replay["realized_under"].to_numpy(int),
        )
        final.update({"intercept": float(m.intercept_[0]), "slope": float(m.coef_[0, 0])})
    elif chosen == "isotonic":
        m = IsotonicRegression(out_of_bounds="clip").fit(
            replay["raw_under"], replay["realized_under"]
        )
        final.update({
            "x_thresholds": [float(x) for x in m.X_thresholds_],
            "y_thresholds": [float(x) for x in m.y_thresholds_],
        })

    payload = {
        "version": "i2-vnext-full-model-calibration-v2",
        "market_inputs_used": False,
        "scope": "2025 normal-starter games; Retrosheet pregame lineups; observed pre-I2 start slot; exact scoreless recursion",
        "event_model_training": "2023-2024 only",
        "hyperparameter_selection": {
            "train": 2023,
            "test": 2024,
            "selected_half_life_days": half_life,
            "selected_C": c,
            "grid": grid,
        },
        "calibration_selection": {
            "train_segment": "first chronological half of 2025",
            "validation_segment": "second chronological half of 2025",
            "candidates": candidates,
            "chosen": chosen,
        },
        "raw_all_2025": metrics(replay["realized_under"], replay["raw_under"]),
        "final_curve": final,
        "replay_games": int(len(replay)),
        "notes": [
            "Final probability calibration is applied once at the full-I2 level.",
            "No market data is used.",
            "Replay excludes I1-to-I2 pitcher changes; opener/bulk handling remains a separate workflow path.",
            "Observed I2 start slot is fixed before I2 and does not use the I2 scoring outcome.",
            "Live inference continues to use the existing I1 start-slot engine.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    replay.to_csv(args.output.with_name("full_model_oos_2025.csv"), index=False)
    print(json.dumps({
        "replay_games": len(replay),
        "chosen": chosen,
        "raw": payload["raw_all_2025"],
        "validation": candidates[chosen]["validation"],
    }, indent=2))


if __name__ == "__main__":
    main()
