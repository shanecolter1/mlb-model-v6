# Public-source notes

The included adapter uses the MLB Stats API schedule endpoint to identify games
and the game-feed endpoint to obtain play-by-play and boxscore information.

The package intentionally stores raw JSON before normalization so every derived
row can be traced back to its source payload.

Baseball Savant remains the operational venue-factor source. The event-vector
model continues to treat the aggregate run factor and component event factors
as separate comparison arms.
