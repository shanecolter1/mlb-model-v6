#!/usr/bin/env python3
"""Hotfix wrapper for strict I2 replay fallback game join.

The original replay's fallback join used a boolean mask from the merged frame to
index the pre-merge frame. With non-contiguous canonical-master indices this can
raise pandas.errors.IndexingError. This wrapper replaces only that join function
while preserving the replay/model logic unchanged.
"""
import pandas as pd
import i2_strict_asof_replay as replay_mod


def join_master_to_games_fixed(master, games):
    m = master.copy().reset_index(drop=True)
    if "game_number" in m.columns:
        m["game_number_join"] = pd.to_numeric(m["game_number"], errors="coerce").fillna(0).astype(int)
    else:
        m = m.sort_values(["game_date", "away_team_code", "home_team_code"]).reset_index(drop=True)
        m["game_number_join"] = m.groupby(["game_date", "away_team_code", "home_team_code"]).cumcount()

    x = m.merge(
        games,
        left_on=["game_date", "away_team_code", "home_team_code", "game_number_join"],
        right_on=["game_date", "away_code", "home_code", "game_number_join"],
        how="left",
        validate="one_to_one",
        sort=False,
    ).reset_index(drop=True)

    miss = x["game_id"].isna()
    if miss.any():
        counts = games.groupby(["game_date", "away_code", "home_code"]).size().rename("n").reset_index()
        unique = games.merge(
            counts[counts["n"] == 1],
            on=["game_date", "away_code", "home_code"],
            how="inner",
            validate="many_to_one",
        )
        fallback_left = x.loc[miss, ["game_date", "away_team_code", "home_team_code"]].copy()
        fallback_left["_x_index"] = fallback_left.index
        fb = fallback_left.merge(
            unique,
            left_on=["game_date", "away_team_code", "home_team_code"],
            right_on=["game_date", "away_code", "home_code"],
            how="left",
            validate="one_to_one",
            sort=False,
        ).set_index("_x_index")
        for c in ["game_id", "away_team_id", "home_team_id"]:
            x.loc[fb.index, c] = fb[c]
    return x


replay_mod.join_master_to_games = join_master_to_games_fixed

if __name__ == "__main__":
    replay_mod.main()
