# Phase 16: opening I2 PA calibration, canonical monthly walk-forward

The 2025 canonical replay uses monthly point-in-time I2 PA models, baseball-only pregame lineups and starters, and prior-season venue profiles. We exported all nine pregame batter-vs-starter event vectors in both halves **before** joining any observed I2 start slots or outcomes. We then joined the first three actual terminal PAs per half from Statcast, required all six expected batter identities to agree to identify a game, and scored only halves whose first three PAs used the pregame starter. The eight exact event classes use the existing event engine. This is a diagnostic of the already inspected 2025 season, not a fresh locked validation or a change to the production model.

| Coverage | Count |
| --- | ---: |
| 2025 games with exact six-batter identity match | 2,422 / 2,430 |
| Top halves with three starter PAs | 2,373 |
| Bottom halves with three starter PAs | 2,377 |
| Games with both eligible halves | 2,329 |
| Top / bottom halves excluded for pitcher change | 49 / 45 |

| First three PAs | Expected reach | Actual reach | Actual minus expected |
| --- | ---: | ---: | ---: |
| Top (7,119 PAs) | 30.76% | 29.29% | -1.47 pp |
| Bottom (7,131 PAs) | 30.75% | 31.54% | +0.79 pp |

Within the 2,329 paired games, bottom minus top reach residual is **+2.25 pp**, normal game-paired 95% interval **+0.69 to +3.81 pp**. For matched prior-season venues, the top residual is -2.02 pp and bottom +0.75 pp. For the small neutral venue group (486 top and 480 bottom PAs), the top residual is +6.01 pp and bottom +1.36 pp. The neutral subset includes temporary sites with no usable prior-season profile; it does not identify a causal venue familiarity effect. This is a half/venue calibration finding, not proof that park identity or the home batting side alone causes it.

A narrow shadow probe moved probability mass between reach and out while retaining the existing event model and each class's within-group share. Its two side-specific logit offsets were fitted on matched-venue PAs through June 2025 and scored on July onward matched-venue PAs. The fitted top/bottom offsets were -0.08479 / -0.00675. On the 6,417 later PAs, combined multiclass log loss moved from 1.469641 to 1.469108. Top moved 1.454230 to 1.453041; bottom moved 1.484981 to **1.485101**. Bottom actual reach in this later segment was 32.37% against 30.67% expected; the fitted correction moved expectation further down to 30.53%. A constant side offset learned earlier in the season therefore does **not** repair the bottom miss. We do not add it to the engine or claim an I2 Under improvement.

Next concrete model work: determine whether the changing bottom residual is explained by the existing pregame batter/pitcher talent inputs and starter-continuation assumptions across season segments, using the exact PA and half audit together. Only implement a pregame feature in the shared event engine if a time-ordered evaluation improves both event calibration and full I2 forecasts. Keep 57.1414% canonical monthly replay and 56.6292% static replay distinct; neither rate changed here.

Reproduce with `node src/research/export_i2_opening_pa_vectors.mjs` and `python src/research/audit_i2_opening_pa_walkforward.py`. The vector export is scratch intermediate data; aggregate results are in `data/derived/i2_vnext/phase16_opening_pa_walkforward_audit.json`.
