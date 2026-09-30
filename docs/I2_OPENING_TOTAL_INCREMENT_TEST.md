# Opening full-game total: paired I2 research test

Research checkpoint, 2026-09-30. No production or vNext pre-freeze input changed.

**Interpretation update:** This is a transfer test of v0.4's fixed formula, **not** a vNext-trained total effect. It should not be used to reject total information for vNext. The subsequently fitted vNext-specific top/bottom test is in `docs/I2_VNEXT_TRAINED_OPENING_TOTAL_TEST.md`.

## Decision and scope

The promoted v0.4 workflow already conditions on a DraftKings pregame full-game total. The newer I2 vNext replay is intentionally baseball-only before the forecast freeze. This experiment applies the *same exact-bucket logit recentering* used by v0.4 **after the archived vNext predictions are fixed**, strictly as a diagnostic. It neither uses an I2 price nor modifies a live probability artifact.

Source: the versioned `historical-mlb-2021-2025-v1` GitHub Release game-level master, SHA-256 of compressed CSV `8512a4fc2fc8e8566d62080bcf006ab6dbdc953e95c3eb8725b55c47a8dafee5`. The repository's master manifest is stale (`pending_release_asset_upload`), so the script records the verified asset digest independently. The master contains DraftKings *opening* full-game total points and reconciled Retrosheet identifiers. Only the total point—not its juice, I2 odds, or final game runs—is used in the transform. Final scores are used solely to verify exact identity when joining the historical v0.4 validation release.

For each test season, the broad and exact-total I2 priors are rebuilt from earlier seasons in the same pitch-clock/rule regime. This is a fixed application of the v0.4 formula, with no coefficient chosen on the test season:

| Test | Prior seasons | Regime |
|---|---|---|
| 2022 | 2021 | Pre-2023 rule environment |
| 2024 | 2023 | Post-2023 rule environment |
| 2025 | 2023–24 | Post-2023 rule environment |

There is no earlier post-rule season to build a comparable 2023 prior. The canonical opening-total archive ends 2025-08-16; 2025 results below cover only matched games through then. The already-inspected 2026 season is excluded from selection and this historical result.

## Paired results (exact same games in each row)

Log loss is lower when better. Day-cluster intervals refer to conditioned minus raw full-I2 log loss.

| Season | Games | Raw full-I2 | With total | Difference | 95% day-cluster interval |
|---|---:|---:|---:|---:|---:|
| 2022 | 2,293 | .680577 | .684086 | +.003509 | [−.000416, +.007528] |
| 2024 | 2,354 | .685514 | .688124 | +.002610 | [−.000313, +.005478] |
| 2025 through Aug 16 | 1,659 | .680863 | .680635 | −.000228 | [−.002720, +.002159] |

The half-inning log-loss changes (conditioned minus raw) were top/bottom: 2022 +.001775/+.003187; 2024 +.001232/+.001875; 2025 −.000291/−.000071. The full-I2 and half-inning Brier scores follow the same direction. This *full-strength v0.4 conditioning transform* did not reliably improve vNext; it worsened the 2022 and 2024 point estimates and barely improved 2025. The intervals span zero. These results do not rule out a differently fitted, partial run-environment term, nor do they establish that vNext is superior to production.

## Historical v0.4 reference on matched games

The `i2-v0.4-production-validation-v1` release supplies game-level **historical OOS candidate predictions** (verified release SHA-256 `a10284015798da6e797abca00b6df81e9c16cd72e30c7928e22860672b49a820`). MLB Stats API schedule identifiers were reconciled to the Retrosheet game IDs by official date, teams and final score; ambiguous matches are excluded. This is not a timestamped archive of frozen live production forecasts, and its model/training differs from vNext. The following same-game reference therefore cannot settle a prospective production comparison:

| Season | Matched games | vNext raw | vNext + total | v0.4 Local-CV candidate |
|---|---:|---:|---:|---:|
| 2022 | 2,278 | .681024 | .684535 | .679011 |
| 2024 | 2,339 | .685389 | .688108 | .685987 |
| 2025 through Aug 16 | 1,646 | .680448 | .680047 | .680703 |

The comparison is log loss of full-I2 Under 0.5. These seasons and the candidate's design have been inspected in earlier research; they are not pristine holdouts. In particular, do not pool the pre-rule 2022 season with 2024–25 to choose a production rule. The operational decision remains **no promotion**. A final production claim requires paired forecasts frozen before first pitch for a new evaluation period, with the opening total captured before the freeze and I2 prices arriving afterward.
