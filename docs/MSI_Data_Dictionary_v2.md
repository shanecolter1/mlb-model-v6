# MSI Data Dictionary v2.0 — Environmental Context revision

The legacy `park_score` is deprecated. A valid Savant venue profile replaces it.
`weather_score` remains separate until Phase 2 validates park-weather interactions.

| Category | Field | Status | Purpose |
|---|---|---|---|
| Context | venue_run_factor | Active shadow | Baseline venue scoring multiplier |
| Context | venue_hr_factor | Active shadow | HR event-rate and scoring-tail adjustment |
| Context | venue_1b_factor | Active shadow | Single event-rate adjustment |
| Context | venue_2b_factor | Active shadow | Double event-rate adjustment |
| Context | venue_3b_factor | Active shadow | Triple event-rate adjustment |
| Context | venue_lhb_profile | Active shadow | Lineup-weighted LHB venue effects |
| Context | venue_rhb_profile | Active shadow | Lineup-weighted RHB venue effects |
| Context | venue_profile_confidence | Active shadow | Reliability contribution to existing MCI |
| Context | weather_score | Retained | Current-game weather adjustment |
| Context | park_score | Deprecated/fallback | Disabled when Savant profile is valid |
| Audit | venue_source_window | Required | Exact historical window used |
| Audit | venue_profile_snapshot_time | Required | Reproducibility timestamp |
| Audit | generic_park_score_disabled | Required | Double-counting control |
| Audit | venue_fallback_reason | Required when degraded | Graceful-degradation explanation |
