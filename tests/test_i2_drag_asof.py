import unittest
from datetime import date

from src.research.build_i2_drag_asof import asof, extract


class DragAsOfTest(unittest.TestCase):
    def test_extract_and_lagged_weighted_window(self):
        html = ('const serverVals = {"scatterData":['
                '{"game_date":"2026-04-01T00:00:00.000Z","year":2026,'
                '"mean_cd":0.34,"num_pitches":100,"num_games":1},'
                '{"game_date":"2026-04-02T00:00:00.000Z","year":2026,'
                '"mean_cd":0.36,"num_pitches":300,"num_games":2},'
                '{"game_date":"2026-04-03T00:00:00.000Z","year":2026,'
                '"mean_cd":0.31,"num_pitches":200,"num_games":1}]};')
        rows = extract(html)
        early, late = asof(rows, [date(2026, 4, 3), date(2026, 4, 5)], 2)
        self.assertEqual(early["drag_asof_date"], "2026-04-01")
        self.assertEqual(early["drag_pitches_28d"], 100)
        self.assertEqual(late["drag_asof_date"], "2026-04-03")
        self.assertAlmostEqual(late["drag_cd_28d"], (34+108+62)/600)


if __name__ == "__main__":
    unittest.main()
