"""The league environment must exclude every same-day outcome."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "research"))
from build_i2_league_asof_features import build_features, feed_games


class LeagueAsOfFeaturesTest(unittest.TestCase):
    def test_all_innings_and_same_day_gate(self):
        fixture = json.loads((Path(__file__).parent /
                              "fixtures/mlb_statsapi/feeds/1.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for pk, day, events in (
                (1, "2026-04-01", ("home_run", "strikeout")),
                (2, "2026-04-02", ("walk",)),
                (3, "2026-04-02", ("double",)),
            ):
                feed = json.loads(json.dumps(fixture))
                feed["gamePk"] = pk
                feed["gameData"]["game"] = {"type": "R"}
                feed["gameData"]["status"]["abstractGameState"] = "Final"
                feed["gameData"]["datetime"]["officialDate"] = day
                play = feed["liveData"]["plays"]["allPlays"][0]
                feed["liveData"]["plays"]["allPlays"] = []
                for inning, event in enumerate(events, start=1):
                    row = json.loads(json.dumps(play))
                    row["about"]["inning"] = inning
                    row["result"]["eventType"] = event
                    feed["liveData"]["plays"]["allPlays"].append(row)
                path = root / f"{pk}.json"
                path.write_text(json.dumps(feed))
                records.append({"game_id": pk, "game_date": day,
                                "feed_path": str(path)})
            (root / "fetch_manifest.json").write_text(json.dumps({"games": records}))
            daily, games = feed_games(root)
            features = build_features(daily, (14,))
        self.assertEqual(games, 3)
        self.assertEqual(int(features.loc[0, "prior_14d_pa"]), 0)
        self.assertEqual(int(features.loc[1, "prior_14d_pa"]), 2)
        self.assertEqual(int(features.loc[1, "prior_14d_home_run_count"]), 1)
        self.assertEqual(int(features.loc[1, "prior_14d_strikeout_count"]), 1)
        self.assertEqual(int(features.loc[1, "prior_14d_walk_count"]), 0)
        self.assertEqual(int(features.loc[1, "prior_14d_double_count"]), 0)


if __name__ == "__main__":
    unittest.main()
