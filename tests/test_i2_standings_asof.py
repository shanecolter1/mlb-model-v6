"""Standings rows must identify the prior calendar day only."""
import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "research"))
from fetch_i2_standings_asof import parse_snapshot


class StandingsAsOfTest(unittest.TestCase):
    def test_prior_date_and_team_coverage(self):
        payload = {"records": [{"division": {"id": i}, "teamRecords": [
            {"team": {"id": 100 + i * 5 + j}, "wins": 20, "losses": 14,
             "wildCardGamesBack": "2.5", "divisionRank": str(j + 1)}
            for j in range(5)]} for i in range(6)]}
        rows = parse_snapshot(payload, date(2026, 5, 1))
        self.assertEqual(len(rows), 30)
        self.assertTrue(all(r["game_date"] == "2026-05-01" and
                            r["standings_asof_date"] == "2026-04-30" for r in rows))
        self.assertEqual(rows[0]["wild_card_games_back"], "2.5")
        with self.assertRaises(ValueError):
            parse_snapshot({"records": payload["records"][:-1]}, date(2026, 5, 1))


if __name__ == "__main__":
    unittest.main()
