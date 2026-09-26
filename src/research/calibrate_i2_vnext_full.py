#!/usr/bin/env python3
"""Estimate the single final full-I2 calibration curve for vNext.

Chronology:
- 2024 is already reserved for PA-model hyperparameter selection.
- Train through 2024 -> generate 2025 full-I2 OOS probabilities -> fit sigmoid.
- Train through 2025 -> generate 2026 full-I2 OOS probabilities -> validate sigmoid.
- Adopt sigmoid only if it improves both Brier and log loss in 2026; otherwise identity.

The historical replay is intentionally lean. It uses actual inning-two batter order
observed in Statcast for the known batters in that half inning, fills only unobserved
future lineup slots with a league-average hitter, and computes scoreless probability
exactly from the existing empirical event/base/out transition table.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from fit_i2_vnext import add_arsenal_feature, fit_one

EVENTS = [
    "single", "double", "triple", "home_run", "walk", "hit_by_pitch",
    "strikeout", "ball_in_play_out",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("data/derived/i2_vnext/i2_pa_statcast.csv"))
    p.add_argument("--arsenal-dir", type=Path, default=Path("data/derived/i2_vnext/arsenal"))
    p.add_argument("--event-model", type=Path, default=Path("data/derived/i2_vnext/i2_vnext_event_model.json"))
    p.add_argument("--transitions", type=Path, default=Path("data/derived/i2/i2_play_calibration.json"))
    p.add_argument("--output", type=Path, default=Path("data/derived/i2_vnext/i2_vnext_full_calibration.json"))
    return p.parse_args()


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def logit(p: float) -> float:
    p = min(1 - 1e-9, max(1e-9, float(p)))
    return math.log(p / (1 - p))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, dtype=int)
    return {
        "n": int(len(y)),
        "base_rate": float(y.mean()) if len(y) else None,
        "predicted_mean": float(p.mean()) if len(p) else None,
        "brier": float(np.mean((p - y) ** 2)) if len(y) else None,
        "logloss": float(log_loss(y, p, labels=[0, 1])) if len(y) else None,
    }


def bases_to_mask(first: bool, second: bool, third: bool) -> int:
    return int(first) + 2 * int(second) + 4 * int(third)


def fallback_transitions(event: str, outs: int, mask: int):
    first, second, third = bool(mask & 1), bool(mask & 2), bool(mask & 4)
    if event in ("strikeout", "ball_in_play_out"):
        return [(1, mask, 0, 1.0)]
    if event in ("walk", "hit_by_pitch"):
        runs = int(first and second and third)
        post = bases_to_mask(True, first, bool(third or (first and second)))
        return [(0, post, runs, 1.0)]
    if event == "single":
        runs = int(third)
        post = bases_to_mask(True, first, second)
        return [(0, post, runs, 1.0)]
    if event == "double":
        runs = int(second) + int(third)
        post = bases_to_mask(False, True, first)
        return [(0, post, runs, 1.0)]
    if event == "triple":
        return [(0, 4, int(first) + int(second) + int(third), 1.0)]
    if event == "home_run":
        return [(0, 0, 1 + int(first) + int(second) + int(third), 1.0)]
    raise ValueError(event)


def load_transitions(path: Path):
    payload = json.loads(path.read_text())
    raw = payload["base_transitions"]
    out = {}
    for key, rows in raw.items():
        total = sum(float(r.get("p", 0)) for r in rows)
        if total <= 0:
            continue
        out[key] = [
            (
                int(r.get("outs_added", 0)),
                int(r.get("post_mask", 0)),
                int(r.get("runs", 0)),
                float(r.get("p", 0)) / total,
            )
            for r in rows
        ]
    return out


def scoreless_probability(vectors: list[dict[str, float]], transitions: dict) -> float:
    if len(vectors) != 9:
        raise ValueError("Need nine lineup vectors")
    v = np.ones((3, 8, 9), dtype=float)
    for _ in range(1000):
        nxt = np.zeros_like(v)
        for outs in range(3):
            for mask in range(8):
                for slot in range(9):
                    total = 0.0
                    vec = vectors[slot]
                    next_slot = (slot + 1) % 9
                    for event in EVENTS:
                        pe = float(vec.get(event, 0.0))
                        if pe <= 0:
                            continue
                        rows = transitions.get(f"{event}|{outs}|{mask}")
                        if not rows:
                            rows = fallback_transitions(event, outs, mask)
                        for outs_added, post_mask, runs, pt in rows:
                            if runs > 0:
                                continuation = 0.0
                            elif outs + outs_added >= 3:
                                continuation = 1.0
                            else:
                                continuation = v[outs + outs_added, post_mask, next_slot]
                            total += pe * pt * continuation
                    nxt[outs, mask, slot] = total
        delta = float(np.max(np.abs(nxt - v)))
        v = nxt
        if delta < 1e-12:
            break
    return float(v[0, 0, 0])


def predict_vectors(prep, model, feature_rows: pd.DataFrame) -> list[dict[str, float]]:
    p = model.predict_proba(
        prep.transform(
            feature_rows[["batter", "pitcher", "platoon", "arsenal_matchup_xwoba"]]
        )
    )
    classes = [str(x) for x in model.classes_]
    out = []
    for row in p:
        vec = {e: 0.0 for e in EVENTS}
        for cls, value in zip(classes, row):
            if cls in vec:
                vec[cls] = float(value)
        s = sum(vec.values())
        out.append({k: v / s for k, v in vec.items()})
    return out


def league_unknown_vector(
    prep, model, pitcher: str, throws: str, arsenal_matchup_xwoba: float
) -> dict[str, float]:
    rows = pd.DataFrame([
        {
            "batter": "__UNKNOWN__",
            "pitcher": pitcher,
            "platoon": f"Lv{throws}",
            "arsenal_matchup_xwoba": arsenal_matchup_xwoba,
        },
        {
            "batter": "__UNKNOWN__",
            "pitcher": pitcher,
            "platoon": f"Rv{throws}",
            "arsenal_matchup_xwoba": arsenal_matchup_xwoba,
        },
    ])
    vectors = predict_vectors(prep, model, rows)
    return {e: 0.5 * (vectors[0][e] + vectors[1][e]) for e in EVENTS}


def prepare_dataset(path: Path, arsenal_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df[df["game_date"].notna() & df["event_class"].isin(EVENTS)].copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    numeric = df.copy()
    numeric["batter"] = pd.to_numeric(numeric["batter"], errors="raise").astype(int)
    numeric["pitcher"] = pd.to_numeric(numeric["pitcher"], errors="raise").astype(int)
    numeric = add_arsenal_feature(numeric, arsenal_dir)
    df["arsenal_matchup_xwoba"] = numeric["arsenal_matchup_xwoba"].to_numpy()
    df["batter"] = pd.to_numeric(df["batter"], errors="raise").astype(int).astype(str)
    df["pitcher"] = pd.to_numeric(df["pitcher"], errors="raise").astype(int).astype(str)
    df["platoon"] = df["platoon"].fillna("?v?").astype(str)
    if "inning_topbot" not in df.columns:
        raise RuntimeError("Dataset lacks inning_topbot; rebuild with current builder")
    return df


def half_predictions(test: pd.DataFrame, prep, model, transitions: dict) -> pd.DataFrame:
    rows = []
    for (game_pk, half), g in test.groupby(["game_pk", "inning_topbot"], sort=False):
        g = g.sort_values("at_bat_number")
        if g.empty:
            continue
        first_pitcher = str(g.iloc[0]["pitcher"])
        throws = str(g.iloc[0].get("p_throws", "?"))

        known = []
        seen = set()
        for r in g.itertuples(index=False):
            batter = str(r.batter)
            if batter in seen:
                continue
            seen.add(batter)
            known.append({
                "batter": batter,
                "pitcher": first_pitcher,
                "platoon": f"{getattr(r, 'stand', '?')}v{throws}",
                "arsenal_matchup_xwoba": float(r.arsenal_matchup_xwoba),
            })
            if len(known) == 9:
                break

        known_df = pd.DataFrame(known)
        vectors = predict_vectors(prep, model, known_df) if len(known_df) else []
        avg_matchup = (
            float(g["arsenal_matchup_xwoba"].mean()) if len(g) else 0.320
        )
        filler = league_unknown_vector(
            prep, model, first_pitcher, throws, avg_matchup
        )
        while len(vectors) < 9:
            vectors.append(filler)

        p0 = scoreless_probability(vectors[:9], transitions)
        first_score = float(g.iloc[0].get("bat_score", 0) or 0)
        last_post = float(
            g.iloc[-1].get("post_bat_score", first_score) or first_score
        )
        actual_runs = max(0.0, last_post - first_score)
        rows.append({
            "game_pk": int(game_pk),
            "half": str(half),
            "p0": p0,
            "actual_scoreless": int(actual_runs == 0),
        })
    return pd.DataFrame(rows)


def full_game_predictions(halves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for game_pk, g in halves.groupby("game_pk"):
        if len(g) < 2 or g["half"].nunique() < 2:
            continue
        p = float(np.prod(g["p0"].iloc[:2]))
        y = int(np.prod(g["actual_scoreless"].iloc[:2]))
        rows.append({
            "game_pk": int(game_pk),
            "raw_under": p,
            "actual_under": y,
        })
    return pd.DataFrame(rows)


def make_oos_full(
    df: pd.DataFrame,
    train_through: int,
    test_year: int,
    c: float,
    half_life: float,
    transitions: dict,
):
    train = df[df["season"] <= train_through].copy()
    test = df[df["season"] == test_year].copy()
    if train.empty or test.empty:
        raise RuntimeError(
            f"Missing train/test data for {train_through}->{test_year}"
        )
    prep, model = fit_one(train, c, half_life)
    halves = half_predictions(test, prep, model, transitions)
    return full_game_predictions(halves)


def fit_sigmoid(cal: pd.DataFrame):
    x = np.array([[logit(p)] for p in cal["raw_under"]], dtype=float)
    y = cal["actual_under"].astype(int).to_numpy()
    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(x, y)
    return float(model.intercept_[0]), float(model.coef_[0][0])


def apply_sigmoid(
    p: np.ndarray, intercept: float, slope: float
) -> np.ndarray:
    return np.array(
        [sigmoid(intercept + slope * logit(x)) for x in p], dtype=float
    )


def main():
    args = parse_args()
    artifact = json.loads(args.event_model.read_text())
    selected = artifact["selected"]
    c = float(selected["C"])
    half_life = float(selected["half_life_days"])

    df = prepare_dataset(args.dataset, args.arsenal_dir)
    transitions = load_transitions(args.transitions)

    cal_2025 = make_oos_full(
        df, 2024, 2025, c, half_life, transitions
    )
    val_2026 = make_oos_full(
        df, 2025, 2026, c, half_life, transitions
    )
    if len(cal_2025) < 100 or len(val_2026) < 100:
        raise RuntimeError(
            f"Insufficient calibration/validation games: "
            f"{len(cal_2025)}, {len(val_2026)}"
        )

    intercept, slope = fit_sigmoid(cal_2025)
    raw_val = val_2026["raw_under"].to_numpy(dtype=float)
    y_val = val_2026["actual_under"].to_numpy(dtype=int)
    cal_val = apply_sigmoid(raw_val, intercept, slope)

    raw_metrics = metrics(y_val, raw_val)
    calibrated_metrics = metrics(y_val, cal_val)
    adopt = (
        calibrated_metrics["brier"] < raw_metrics["brier"]
        and calibrated_metrics["logloss"] < raw_metrics["logloss"]
    )

    payload = {
        "version": "i2-vnext-full-calibration-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market_inputs_used": False,
        "target": "full I2 Under 0.5 probability",
        "chronology": {
            "hyperparameter_selection": 2024,
            "calibration_fit": 2025,
            "calibration_validation": 2026,
        },
        "calibration_fit_games": int(len(cal_2025)),
        "validation_games": int(len(val_2026)),
        "candidate": {
            "method": "sigmoid_logit",
            "intercept": intercept,
            "slope": slope,
        },
        "validation": {
            "raw": raw_metrics,
            "sigmoid": calibrated_metrics,
            "delta_brier": (
                calibrated_metrics["brier"] - raw_metrics["brier"]
            ),
            "delta_logloss": (
                calibrated_metrics["logloss"] - raw_metrics["logloss"]
            ),
        },
        "selected": (
            {
                "method": "sigmoid_logit",
                "intercept": intercept,
                "slope": slope,
            }
            if adopt
            else {
                "method": "identity",
                "intercept": 0.0,
                "slope": 1.0,
            }
        ),
        "selection_rule": (
            "sigmoid must improve both 2026 Brier and log loss; "
            "otherwise identity"
        ),
        "historical_replay_limitations": [
            "Uses known I2 batter order for batters actually observed in the half inning; unobserved future lineup slots are league-average fillers.",
            "Historical replay does not apply venue park factors; park remains a live baseball input before final calibration.",
            "Actual scoreless label uses terminal-PA bat_score/post_bat_score and may miss rare runs scored before the first terminal PA on a nonterminal pitch.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
