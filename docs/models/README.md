# CFB Model Reports

Per-model metrics, calibration, and provenance.

## Models

- [CPOE](cpoe.md) — The Completion Percentage Over Expected (CPOE) model estimates the probability a given pass attempt is completed (`cp`) from pre-throw game state.
- [Expected Points (EP)](ep.md) — The Expected Points (EP) model estimates the expected next-score value for the team in possession at the **start of a play**, given game state.
- [Fourth-Down Yards](fourth_down.md) — The fourth-down yards model predicts the **distribution of yards gained** on a go-for-it (or third-down) attempt, which feeds the fourth-down decision surface (go / punt / field-goal expected-value comparison).
- [Field Goal (make prob)](fg.md) — The field-goal model estimates the probability a placekick is **made**, given only the kick distance.
- [Pregame WP (Five Factors)](pregame_wp.md) — The pregame Win Probability model (Track 4, the **Five Factors** surface) forecasts a matchup's outcome from a single composite team-quality signal.
- [QBR](qbr.md) — The QBR model reconstructs an ESPN-Total-QBR-style 0-100 quarterback rating from EPA components, so a QBR can be produced for any game in the corpus without an ESPN QBR feed.
- [Two-Point Conversion](two_pt.md) — The two-point-conversion model estimates the probability a two-point attempt **succeeds**, given game context.
- [Win Probability (naive)](wp_naive.md) — The naive Win Probability model answers *given only the game state, with no betting-market information, how likely is the possession team to win?* It is the spread model's sibling — identical except it drops the spread signal — and is the right surface when a pregame spread is unavailable or when you explicitly want a market-free WP.
- [Win Probability (spread)](wp_spread.md) — The spread-aware Win Probability model estimates the probability that the team in possession wins the game, given game state **and the pregame point spread**.
- [Expected Pass](xpass.md) — The expected-pass model estimates the probability that a scrimmage play is a **dropback (pass)** given pre-snap game state — a measure of how *predictable* an offense's tendency is in a given situation.
- [RB Evaluation (xREPA)](rb_eval.md) — The RB-evaluation (xREPA) model maps a running back's per-play efficiency to an **expected** rushing EPA, isolating the back's contribution from team/context.
