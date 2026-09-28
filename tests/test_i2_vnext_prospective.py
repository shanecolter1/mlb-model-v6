import json
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "research"))
from i2_vnext_prospective import archive, fetch_final_feeds, metrics, official_outcome, score


def snapshot(
    generated,
    probability,
    prediction_class="CONFIRMED_INPUTS",
    invalidated=False,
    *,
    pk=123,
    date="2026-09-27",
    game_date="2026-09-27T18:00:00Z",
    validation_start="2026-09-26",
    half_version=None,
    half_h=0.0757053177820764,
    venue_matched=True,
):
    game = {
        "gamePk": pk,
        "gameDate": game_date,
        "modelStatus": "FROZEN_VNEXT_SHADOW_PROJECTION",
        "predictionClass": prediction_class,
        "venueStatus": "SAVANT_PROFILE_MATCHED" if venue_matched else "SAVANT_PROFILE_UNMATCHED_NEUTRAL_FALLBACK",
        "under05": probability,
        "over05": 1-probability,
        "top2ScorePct": 20,
        "bottom2ScorePct": 25,
        "bettingEligibility": {"eligible": False},
        "inputAudit": {"freezeCheck": {
            "checkedAt": generated,
            "gate": {"requiresCleanRerun": invalidated},
        }},
    }
    payload = {
        "date": date,
        "generatedAt": generated,
        "cutoff": generated,
        "marketDataUsed": False,
        "promotionStatus": "SHADOW_ONLY_PROSPECTIVE_VALIDATION_REQUIRED",
        "prospectiveValidationStart": validation_start,
        "games": [game],
    }
    if half_version:
        payload["halfCalibration"] = {
            "version": half_version,
            "type": "zero_sum_logit_contrast",
            "h": half_h,
            "matchedHomeVenueOnly": True,
        }
        payload["finalCalibration"] = {"type": "none", "intercept": 0, "slope": 1}
        game.update({
            "dataAudit": {"venueProfileMatched": venue_matched},
            "halfContrastApplied": venue_matched,
            "halfContrastH": half_h if venue_matched else 0,
            "rawTop2ScoreProbability": .21,
            "rawBottom2ScoreProbability": .24,
            "top2ScoreProbability": .20 if venue_matched else .21,
            "bottom2ScoreProbability": .25 if venue_matched else .24,
            "rawTop2ScorePct": 21,
            "rawBottom2ScorePct": 24,
            "halfAdjustedUnder05": .60 if venue_matched else .6004,
            "rawUnder05": .6004,
        })
    return payload


def final_feed(pk, top, bottom, state="Final"):
    return {
        "gamePk": pk,
        "gameData": {"status": {"abstractGameState": state}},
        "liveData": {"linescore": {"innings": [
            {"num": 1, "away": {"runs": 0}, "home": {"runs": 0}},
            {"num": 2, "away": {"runs": top}, "home": {"runs": bottom}},
        ]}},
    }


class ProspectiveTest(unittest.TestCase):
    def test_auc_counts_ties_and_calibration_bins(self):
        rows = [{"p": .8, "y": 1}, {"p": .4, "y": 0}, {"p": .4, "y": 1}]
        result = metrics(rows, "p", "y")
        self.assertEqual(result["auc"], .75)
        self.assertEqual(sum(bucket["n"] for bucket in result["calibration_bins"]), 3)

    def test_archive_and_score_choose_latest_valid_pregame_snapshot(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            archive_dir, outcome_dir = root / "archive", root / "outcomes"
            outcome_dir.mkdir()
            for index, (generated, p, kind, invalidated) in enumerate([
                ("2026-09-27T16:00:00Z", .60, "CONFIRMED_INPUTS", False),
                ("2026-09-27T17:00:00Z", .62, "PROVISIONAL_EXPECTED_INPUTS", False),
                ("2026-09-27T17:30:00Z", .90, "CONFIRMED_INPUTS", True),
            ]):
                file = root / f"snapshot_{index}.json"
                file.write_text(json.dumps(snapshot(generated, p, kind, invalidated)))
                archived = archive(file, archive_dir)
                self.assertEqual(archive(file, archive_dir), archived)
            (outcome_dir / "123.json").write_text(json.dumps(final_feed(123, 0, 0)))
            prior = root / "prior.csv"
            prior.write_text("season,gid,half,i2_runs\n2025,A,top,0\n2025,A,bottom,0\n2025,B,top,1\n2025,B,bottom,0\n")
            report = score(archive_dir, outcome_dir, prior)
            self.assertEqual(report["archive_snapshots"], 3)
            self.assertEqual(report["scored_games"], 1)
            self.assertEqual(report["scored_rows"][0]["p_under"], .62)
            self.assertEqual(report["scored_rows"][0]["prediction_class"], "PROVISIONAL_EXPECTED_INPUTS")
            self.assertEqual(report["baseline"]["probability"], .5)
            self.assertIsNone(report["vnext_minus_prior"]["ci95_date_cluster"])

    def test_score_isolates_latest_validation_cohort(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            archive_dir, outcome_dir = root / "archive", root / "outcomes"
            outcome_dir.mkdir()
            old = root / "old.json"
            old.write_text(json.dumps(snapshot(
                "2026-09-27T16:00:00Z", .61,
                pk=123,
                date="2026-09-29",
                game_date="2026-09-29T18:00:00Z",
                validation_start="2026-09-26",
            )))
            new = root / "new.json"
            new.write_text(json.dumps(snapshot(
                "2026-09-28T16:00:00Z", .60,
                pk=124,
                date="2026-09-29",
                game_date="2026-09-29T21:00:00Z",
                validation_start="2026-09-28",
                half_version="i2-vnext-half-contrast-v1",
            )))
            archive(old, archive_dir)
            archive(new, archive_dir)
            (outcome_dir / "123.json").write_text(json.dumps(final_feed(123, 0, 0)))
            (outcome_dir / "124.json").write_text(json.dumps(final_feed(124, 0, 1)))
            prior = root / "prior.csv"
            prior.write_text("season,gid,half,i2_runs\n2025,A,top,0\n2025,A,bottom,0\n2025,B,top,1\n2025,B,bottom,0\n")

            report = score(archive_dir, outcome_dir, prior)
            self.assertEqual(report["version"], "i2-vnext-prospective-score-v2")
            self.assertEqual(report["all_selected_games_before_cohort_filter"], 2)
            self.assertEqual(report["excluded_prior_cohort_games"], 1)
            self.assertEqual(report["selected_games"], 1)
            self.assertEqual(report["scored_games"], 1)
            self.assertEqual(report["scored_rows"][0]["game_pk"], 124)
            self.assertEqual(
                report["validation_cohort"]["half_calibration_version"],
                "i2-vnext-half-contrast-v1",
            )
            self.assertEqual(report["validation_cohort"]["prospective_validation_start"], "2026-09-28")
            self.assertEqual(report["half_innings"]["top"]["n"], 1)
            self.assertEqual(report["half_innings_raw"]["top"]["n"], 1)
            self.assertEqual(report["half_contrast_adjusted_minus_raw"]["n"], 2)
            self.assertEqual(report["full_i2_half_adjusted_minus_raw"]["n"], 1)

    def test_rejects_multiple_current_calibration_versions_same_start(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            archive_dir = root / "archive"
            for pk, version in ((123, "half-v1"), (124, "half-v2")):
                file = root / f"{pk}.json"
                file.write_text(json.dumps(snapshot(
                    "2026-09-28T16:00:00Z", .60,
                    pk=pk,
                    date="2026-09-29",
                    game_date="2026-09-29T21:00:00Z",
                    validation_start="2026-09-28",
                    half_version=version,
                )))
                archive(file, archive_dir)
            prior = root / "prior.csv"
            prior.write_text("season,gid,half,i2_runs\n2025,A,top,0\n2025,A,bottom,0\n")
            with self.assertRaisesRegex(ValueError, "Multiple current prospective validation cohorts"):
                score(archive_dir, root / "outcomes", prior)

    def test_rejects_post_first_pitch_and_nonfinal_or_wrong_game_outcome(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            file = root / "late.json"
            file.write_text(json.dumps(snapshot("2026-09-27T18:01:00Z", .60)))
            with self.assertRaisesRegex(ValueError, "after first pitch"):
                archive(file, root / "archive")
        self.assertIsNone(official_outcome(final_feed(123, 0, 0, "Live"), 123))
        with self.assertRaisesRegex(ValueError, "game ID mismatch"):
            official_outcome(final_feed(999, 0, 0), 123)

    def test_fetches_each_final_mlb_feed_once_and_preserves_saved_outcome(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            file = root / "snapshot.json"
            file.write_text(json.dumps(snapshot("2026-09-27T17:00:00Z", .62)))
            archive_dir, outcome_dir = root / "archive", root / "outcomes"
            archive(file, archive_dir)
            called = []
            def fake_open(request, timeout):
                called.append((request.full_url, timeout))
                return io.BytesIO(json.dumps(final_feed(123, 0, 1)).encode())
            first = fetch_final_feeds(archive_dir, outcome_dir, fake_open)
            self.assertEqual(first["fetched_final"], [123])
            self.assertEqual(len(called), 1)
            second = fetch_final_feeds(archive_dir, outcome_dir, fake_open)
            self.assertEqual(second["already_saved"], [123])
            self.assertEqual(len(called), 1)


if __name__ == "__main__":
    unittest.main()
