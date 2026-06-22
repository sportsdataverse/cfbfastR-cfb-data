# CFB Model Reports

Per-model metrics, calibration, and provenance.

## Models

- [CPOE](cpoe.md)
- [Expected Points (EP)](ep.md) — The Expected Points (EP) model estimates the expected next-score value for the team in possession at the **start of a play**, given game state.
- [Fourth-Down Yards](fourth_down.md) — The fourth-down yards model predicts the **distribution of yards gained** on a go-for-it (or third-down) attempt, which feeds the fourth-down decision surface (go / punt / field-goal expected-value comparison).
- [QBR](qbr.md) — The QBR model reconstructs an ESPN-Total-QBR-style 0-100 quarterback rating from EPA components, so a QBR can be produced for any game in the corpus without an ESPN QBR feed.
- [Win Probability (naive)](wp_naive.md) — The naive Win Probability model answers *given only the game state, with no betting-market information, how likely is the possession team to win?* It is the spread model's sibling — identical except it drops the spread signal — and is the right surface when a pregame spread is unavailable or when you explicitly want a market-free WP.
- [Win Probability (spread)](wp_spread.md) — The spread-aware Win Probability model estimates the probability that the team in possession wins the game, given game state **and the pregame point spread**.
- [RB Evaluation (xREPA)](rb_eval.md) — The RB-evaluation (xREPA) model maps a running back's per-play efficiency to an **expected** rushing EPA, isolating the back's contribution from team/context.
