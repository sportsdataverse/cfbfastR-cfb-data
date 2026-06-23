"""Per-model prose blocks (summary / recipe / discussion / limitations) keyed by
model_type. These are authored narratives, not generated, and use the lineage
facts established by the model-training program. The CLI looks each model up by
``model_type`` and injects the prose into the enriched ``ModelReport``.

Each value is a ``ModelNarrative`` with four Markdown strings. Models without an
entry fall back to empty prose (the metrics/figures/provenance sections still
render).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelNarrative:
    summary: str
    recipe: str
    discussion: str
    limitations: str
    # Enriched methodology sections (nflfastR-post structure). Optional — default
    # empty so the four legacy sections remain the required contract.
    features: str = ""   # "## Model features" — a table/list describing EACH feature
    model: str = ""      # "## The model" — algorithm, objective, hyperparameters, CV
    importance: str = "" # "## Feature importance" — prose where applicable


NARRATIVES: dict[str, ModelNarrative] = {
    "cpoe": ModelNarrative(
        summary=(
            "The Completion Percentage Over Expected (CPOE) model estimates the probability a "
            "given pass attempt is completed (`cp`) from pre-throw game state. CPOE is the "
            "**percentage-point residual** `100 * (complete_pass - cp)`: positive when a passer "
            "completes throws a league-average passer would not. It is the CFB analogue of the "
            "nflfastR CP/CPOE surface."
        ),
        recipe=(
            "An 8-feature XGBoost **binary:logistic** completion model. Lineage is the nflfastR "
            "CP recipe adapted to CFB game state; the residual `cp` is subtracted from the binary "
            "completion outcome and scaled to percentage points."
        ),
        discussion=(
            "No CPOE leave-one-season-out OOF parquet is shipped in this artifacts set, so a "
            "per-play completion-probability calibration figure is **not** rendered here — the "
            "card is prose + provenance only. When a CPOE OOF (predicted `cp` + actual "
            "`complete_pass` + `period`) is produced, the same cfbscrapR binning recipe used for "
            "WP applies directly: bin `cp` into 0.05 buckets and compare to the empirical "
            "completion rate."
        ),
        limitations=(
            "CPOE is blind to receiver separation, pressure, and air-yards charting we do not "
            "have, so it captures the *game-state-explainable* part of completion probability "
            "only. Without a shipped OOF parquet, the calibration claim here is deferred to the "
            "model card rather than shown out-of-sample."
        ),
        features=(
            "**8 features**, all known before the throw; the binary label is `complete_pass`.\n\n"
            "| Feature | Type | What it encodes |\n"
            "|---|---|---|\n"
            "| `down` | numeric | Current down. |\n"
            "| `distance` | numeric | Yards to go. |\n"
            "| `yards_to_goal` | numeric | Field position. |\n"
            "| `score_diff` | numeric | Possession-team score differential. |\n"
            "| `seconds_remaining` | numeric | Clock context. |\n"
            "| `is_home` | binary | Home-field indicator. |\n"
            "| `period` | numeric | Quarter. |\n"
            "| `passing_down` | binary | Whether the situation is an obvious passing down. |\n"
        ),
        model=(
            "**Algorithm.** XGBoost, `objective=binary:logistic`. The predicted `cp` is the "
            "completion probability; **CPOE = `100 * (complete_pass - cp)`** on a percentage-point "
            "scale.\n\n"
            "**Evaluation.** No season-held-out OOF is shipped here, so calibration is deferred "
            "to the model card (see Provenance). The intended check is the cfbscrapR probability "
            "binning recipe on a CPOE OOF once produced."
        ),
        importance=(
            "Completion probability is driven primarily by `distance` / `yards_to_goal` (throw "
            "depth proxies) and `passing_down`; the clock/score context contributes a smaller "
            "game-script correction."
        ),
    ),
    "ep": ModelNarrative(
        summary=(
            "The Expected Points (EP) model estimates the expected next-score value for the "
            "team in possession at the **start of a play**, given game state. It is the "
            "foundation of the whole CFB analytics stack: EP differences between consecutive "
            "plays define **Expected Points Added (EPA)**, which in turn feeds the QBR model "
            "and the RB-evaluation (xREPA) surface. Downstream, every play-by-play row carries "
            "an `ep` column plus the seven class probabilities."
        ),
        recipe=(
            "An 8-feature XGBoost **multiclass softprob** model over **7 next-score classes** "
            "(touchdown for/against, field goal for/against, safety for/against, no-score). "
            "Class probabilities are collapsed to a single EP via the point-value map "
            "`{0:+7, 1:-7, 2:+3, 3:-3, 4:+2, 5:-2, 6:0}`. The recipe is a faithful port of the "
            "cfbscrapR / `keepers` EP model. Features: `TimeSecsRem`, `yards_to_goal`, "
            "`distance`, one-hot `down_1..down_4`, and `pos_score_diff_start`. Retrained on the "
            "**full 2004-2025 history (2,219,607 plays)** with the shipped hyperparameters "
            "unchanged."
        ),
        discussion=(
            "Metrics are pooled **leave-one-season-out (LOSO)** out-of-fold predictions — for "
            "each season we train on every *other* season and predict the held-out one, so the "
            "numbers are honest out-of-sample. Pooled `mlogloss` 1.2333 and top-1 accuracy "
            "0.4997 are strong for a 7-way score-outcome problem where the modal outcome (no "
            "score / TD-for) is inherently noisy. The headline calibration number is the "
            "**EP calibration MAE of 0.014 points**: binned predicted EP tracks realized "
            "next-score value almost exactly (`mean_pred_EP == mean_realized == 1.689`). The "
            "calibration figure plots binned predicted EP against realized next-score value "
            "against the y=x line."
        ),
        limitations=(
            "EP is a *start-of-play* quantity; it does not know the result of the current play "
            "(that is what EPA captures). Top-1 accuracy near 0.50 reflects irreducible outcome "
            "noise, not miscalibration — the model is well-calibrated in aggregate even where "
            "individual-play outcomes are unpredictable. The 7-class point map is fixed (no "
            "2-point-conversion modelling beyond the safety/defensive-score classes), and the "
            "model is blind to weather, personnel, and in-play participants by design."
        ),
        features=(
            "The EP model uses **8 features**, all known at the **start of the play** — no "
            "look-ahead. Each row is one scrimmage play; the label is the *next scoring event* in "
            "the same half.\n\n"
            "| Feature | Type | What it encodes |\n"
            "|---|---|---|\n"
            "| `TimeSecsRem` | numeric | Seconds remaining in the half — late-half plays have "
            "fewer expected possessions left to score. |\n"
            "| `yards_to_goal` | numeric | Distance (1-99) to the opponent's end zone — the single "
            "strongest field-position signal. |\n"
            "| `distance` | numeric | Yards to go for a first down. |\n"
            "| `down_1` … `down_4` | one-hot | Current down, one-hot encoded (4 columns) so the "
            "tree can split cleanly on each down. |\n"
            "| `pos_score_diff_start` | numeric | Possession-team score differential — late-game "
            "score state shifts play-calling and therefore next-score expectation. |\n"
        ),
        model=(
            "**Algorithm.** XGBoost gradient-boosted trees, `objective=multi:softprob` over "
            "`num_class=7`, `eval_metric=mlogloss`. **525 boosting rounds**, `eta=0.025`, "
            "`max_depth=5`, `subsample=0.8`, `colsample_bytree=0.8`, `gamma=1`, "
            "`min_child_weight=1` — the exact cfbscrapR / `keepers` hyperparameters, unchanged. "
            "Rows are weighted by `ScoreDiff_W` (the cfbscrapR score-differential weighting). The "
            "7 class probabilities are dotted with the point map `{0:+7, 1:-7, 2:+3, 3:-3, 4:+2, "
            "5:-2, 6:0}` to produce a scalar EP.\n\n"
            "**Evaluation.** Honest **leave-one-season-out (LOSO)** cross-validation: for each of "
            "the 22 seasons (2004-2025) we retrain on the *other* 21 seasons and predict the "
            "held-out one, then pool the out-of-fold predictions. No play is ever scored by a "
            "model that saw its season in training."
        ),
        importance=(
            "By XGBoost gain, `yards_to_goal` dominates (field position is the backbone of EP), "
            "followed by `TimeSecsRem` and `pos_score_diff_start`; the down one-hots and "
            "`distance` refine the surface within a given field position. This ordering matches "
            "the cfbscrapR EP model and the nflfastR EP post."
        ),
    ),
    "wp_spread": ModelNarrative(
        summary=(
            "The spread-aware Win Probability model estimates the probability that the team in "
            "possession wins the game, given game state **and the pregame point spread**. It "
            "produces the `vegas_wp`-style surface; consecutive-play differences define "
            "**Win Probability Added (WPA)**."
        ),
        recipe=(
            "A 13-feature XGBoost **binary:logistic** model, **760 trees**, a faithful port of "
            "the `cfbscrapR-wpa.ipynb` recipe. The signature feature is "
            "`spread_time = pos_team_spread * exp(-4 * elapsed_share)` — the pregame spread "
            "decayed toward zero as the game clock runs out, so its influence vanishes by the "
            "fourth quarter. Other features: `TimeSecsRem`, `adj_TimeSecsRem`, "
            "`ExpScoreDiff_Time_Ratio`, `pos_score_diff_start`, `down`, `distance`, "
            "`yards_to_goal`, `is_home`, both teams' remaining timeouts, `period`, and "
            "`pos_team_receives_2H_kickoff`."
        ),
        discussion=(
            "LOSO pooled: `logloss` 0.3616, Brier 0.1182, AUC 0.9159, weighted calibration "
            "error 0.0147. The AUC near 0.92 and the sub-0.015 weighted calibration error mean "
            "the predicted win probabilities are both discriminating and well-calibrated across "
            "the whole probability range. The calibration figure facets by quarter so you can "
            "confirm calibration holds late in games (where WP is most actionable)."
        ),
        limitations=(
            "WPA — the first difference of WP — is intrinsically noisy: small per-play WP "
            "movements are dominated by model variance, so single-play WPA should be read as a "
            "directional signal, not a precise quantity. The spread input is a pregame number; "
            "the model does not re-estimate a live spread. Overtime and end-of-half edge cases "
            "are handled by the construction pipeline upstream, not by the model head."
        ),
        features=(
            "**13 features**, all start-of-play. The binary label is `win_indicator = "
            "(possession team == game winner)`. The signature feature is the spread-decay term.\n\n"
            "| Feature | Type | What it encodes |\n"
            "|---|---|---|\n"
            "| `spread_time` | numeric | `pos_team_spread * exp(-4 * elapsed_share)` — the pregame "
            "spread decayed toward 0 as the clock runs; its influence vanishes by Q4. **The "
            "market signal.** |\n"
            "| `TimeSecsRem` | numeric | Seconds remaining in the half. |\n"
            "| `adj_TimeSecsRem` | numeric | Game-clock-adjusted time remaining (half-aware). |\n"
            "| `ExpScoreDiff_Time_Ratio` | numeric | Expected score differential scaled by time — "
            "a momentum/urgency interaction. |\n"
            "| `pos_score_diff_start` | numeric | Possession-team score differential. |\n"
            "| `down` | numeric | Current down. |\n"
            "| `distance` | numeric | Yards to go. |\n"
            "| `yards_to_goal` | numeric | Field position. |\n"
            "| `is_home` | binary | Home-field indicator for the possession team. |\n"
            "| `pos_team_timeouts_rem_before` | numeric | Possession-team timeouts left. |\n"
            "| `def_pos_team_timeouts_rem_before` | numeric | Defense timeouts left. |\n"
            "| `period` | numeric | Quarter (1-4+). |\n"
            "| `pos_team_receives_2H_kickoff` | binary | Whether the possession team gets the "
            "second-half kickoff — a known WP edge. |\n"
        ),
        model=(
            "**Algorithm.** XGBoost, `objective=binary:logistic`, `eval_metric=logloss`, "
            "**760 boosting rounds**, `eta=0.02`, `max_depth=5`, `min_child_weight=14`, "
            "`subsample=0.72`, `colsample_bytree=0.57`, `gamma=0.34` — the exact "
            "`cfbscrapR-wpa.ipynb` recipe, unchanged. No sample weights (per the cfbscrapR WPA "
            "recipe).\n\n"
            "**Evaluation.** Leave-one-season-out over 2004-2025: train on 21 seasons, predict "
            "the held-out one, pool the out-of-fold win probabilities. The calibration figure "
            "**facets by quarter** (the cfbscrapR `03-WPA-Model.R` recipe) so you can confirm "
            "calibration holds late in games, where WP is most actionable."
        ),
        importance=(
            "`spread_time` and the time/score-differential terms carry the model early in games; "
            "as `spread_time` decays, `pos_score_diff_start`, `yards_to_goal` and the clock terms "
            "take over. This is exactly the intended hand-off from market prior to live game "
            "state."
        ),
    ),
    "wp_naive": ModelNarrative(
        summary=(
            "The naive Win Probability model answers *given only the game state, with no "
            "betting-market information, how likely is the possession team to win?* It is the "
            "spread model's sibling — identical except it drops the spread signal — and is the "
            "right surface when a pregame spread is unavailable or when you explicitly want a "
            "market-free WP."
        ),
        recipe=(
            "A 12-feature XGBoost **binary:logistic** model, **65 trees**. The feature set is "
            "exactly the spread model's **minus `spread_time`**; everything else (game clock, "
            "score differential, field position, down/distance, timeouts, period, 2H-kickoff "
            "possession) is shared. Far fewer trees than the spread model (65 vs 760) because "
            "without the spread there is less structured signal to fit."
        ),
        discussion=(
            "Metrics are pooled **leave-one-season-out (LOSO)** out-of-fold predictions (the "
            "naive variant now gets its own LOSO pass, identical in protocol to the spread "
            "model). The naive WP **correlates ~0.94 with the spread WP** and, as expected, the "
            "two **diverge most in the first quarter** — when the pregame spread carries the most "
            "information and the game state carries the least. By the fourth quarter the mean "
            "absolute gap between naive and spread WP collapses (Q1 ~0.13 vs Q4 ~0.02), because "
            "`spread_time` has decayed away and both models are reading the same near-final game "
            "state. The calibration figure facets by quarter, the same cfbscrapR recipe as the "
            "spread model."
        ),
        limitations=(
            "Because it ignores the market, the naive model is *less sharp* early in games: its "
            "log-loss and Brier are worse than the spread model's (it has strictly less "
            "information). It is the correct tool only when you want a spread-free WP or lack a "
            "spread; for forecasting accuracy when a spread exists, prefer the spread model. WPA "
            "(the first difference of WP) carries the same per-play noise caveat as the spread "
            "model."
        ),
        features=(
            "**12 features** — exactly the spread model's set **minus `spread_time`**. Everything "
            "else is shared: `TimeSecsRem`, `adj_TimeSecsRem`, `ExpScoreDiff_Time_Ratio`, "
            "`pos_score_diff_start`, `down`, `distance`, `yards_to_goal`, `is_home`, both teams' "
            "remaining timeouts, `period`, and `pos_team_receives_2H_kickoff`. Dropping the "
            "single market feature is the *only* difference between the two WP heads, which is why "
            "they can be compared head-to-head."
        ),
        model=(
            "**Algorithm.** XGBoost, `objective=binary:logistic`, **65 boosting rounds**, "
            "`eta=0.2`, `max_depth=4`, `subsample=0.8`, `colsample_bytree=0.8` — far fewer trees "
            "than the spread model (65 vs 760) because, without the spread, there is less "
            "structured signal to fit and the model saturates earlier.\n\n"
            "**Evaluation.** Leave-one-season-out over 2004-2025, pooled out-of-fold, faceted by "
            "quarter — the same protocol as the spread model, so the two are directly comparable."
        ),
        importance=(
            "Without the market prior, `pos_score_diff_start`, `yards_to_goal` and the clock "
            "terms carry the model from the opening kickoff; this is precisely why the naive WP "
            "is least confident (closest to 0.5) early and why it diverges most from the spread "
            "WP in Q1."
        ),
    ),
    "qbr": ModelNarrative(
        summary=(
            "The QBR model reconstructs an ESPN-Total-QBR-style 0-100 quarterback rating from "
            "EPA components, so a QBR can be produced for any game in the corpus without an ESPN "
            "QBR feed. It is a per-(quarterback, game) regression onto ESPN's published raw QBR."
        ),
        recipe=(
            "A 6-feature XGBoost regression, **45 trees**, full-history retrain. Features are "
            "the per-game weighted-mean EPA components that drive QBR: `qbr_epa`, `sack_epa`, "
            "`pass_epa`, `rush_epa`, `pen_epa`, plus the posteam `spread`. The EPA components "
            "come from the same EP model documented above, so QBR sits one layer above EP/EPA."
        ),
        discussion=(
            "LOSO pooled over 22,833 quarterback-games (2005-2025): RMSE 17.87, MAE 13.90, "
            "R² 0.585, correlation 0.765. It **decisively beats the legacy 2020 model**: on the "
            "2021-25 holdout, RMSE 16.1 vs 23.2 and R² 0.66 vs 0.29. The earlier (pre-2014) "
            "seasons carry most of the residual error — fold RMSEs run ~19-24 before 2014 and "
            "~16 after — consistent with sparser / noisier early ESPN QBR labels. The scatter "
            "figure plots predicted QBR against ESPN raw QBR with the y=x reference."
        ),
        limitations=(
            "QBR is a **bounded 0-100** target, so an RMSE of ~18 points is large relative to "
            "the scale and the model cannot perfectly reproduce ESPN's proprietary formula "
            "(which uses clutch weighting and charting inputs we do not have). The 2004 fold has "
            "no joined rows (no ESPN QBR labels), and pre-2014 error is materially higher. Treat "
            "the output as a faithful reconstruction of the *EPA-explainable* part of QBR, not a "
            "byte-exact ESPN replica."
        ),
        features=(
            "**6 features**, one row per (quarterback, game). Each EPA component is the per-game "
            "weighted mean of that component over the QB's plays (the same weighting "
            "`CFBPlayProcess.__process_qbr` uses).\n\n"
            "| Feature | Type | What it encodes |\n"
            "|---|---|---|\n"
            "| `qbr_epa` | numeric | Total QBR-attributable EPA per game — the dominant driver. |\n"
            "| `sack_epa` | numeric | EPA lost to sacks. |\n"
            "| `pass_epa` | numeric | EPA from pass attempts. |\n"
            "| `rush_epa` | numeric | EPA from QB rushes. |\n"
            "| `pen_epa` | numeric | EPA from penalties on the QB's plays. |\n"
            "| `spread` | numeric | Possession-team pregame spread (context for garbage-time "
            "deflation). |\n"
        ),
        model=(
            "**Algorithm.** XGBoost regression (squared-error objective), **45 boosting rounds**, "
            "full-history retrain. The target is ESPN's *published raw QBR* for the "
            "quarterback-game; the EPA components come from the EP model documented above, so QBR "
            "sits one layer above EP/EPA.\n\n"
            "**Evaluation.** Leave-one-season-out over 22,833 quarterback-games (2005-2025). On "
            "the 2021-25 holdout it **decisively beats the legacy 2020 model** (RMSE 16.1 vs 23.2, "
            "R² 0.66 vs 0.29). Because QBR is a continuous bounded target, the calibration figure "
            "is a predicted-vs-actual scatter (2-D bin density) with a y=x reference, not a "
            "probability-bucket plot.\n\n"
            "**Rule-era variant (adopted).** Adding the one-hot era dummies (`era0..era3`, cuts "
            "2006/2013/2020) is the one *material* era win in the suite — pooled LOSO RMSE "
            "**17.88 → 17.42** (evaluated on the spread-backfilled frame, since `spread` is a "
            "feature). Shipped side-by-side as `qbr_era.ubj` (10 features)."
        ),
        importance=(
            "`qbr_epa` overwhelmingly dominates by gain (it *is* the EPA aggregate ESPN's QBR "
            "tracks), with `pass_epa` and `rush_epa` next; `spread` contributes a small "
            "garbage-time / leverage correction."
        ),
    ),
    "fourth_down": ModelNarrative(
        summary=(
            "The fourth-down yards model predicts the **distribution of yards gained** on a "
            "go-for-it (or third-down) attempt, which feeds the fourth-down decision surface "
            "(go / punt / field-goal expected-value comparison). From the gain distribution we "
            "derive P(first down) for any distance-to-go."
        ),
        recipe=(
            "A 6-feature XGBoost **multiclass softprob** over **76 classes** (integer gains "
            "-10..65), **157 rounds**. Lineage is the cfb4th model plus an added **ordinal CFB "
            "rule-era factor** `era` (0:&le;2006, 1:2007-13, 2:2014-17, 3:&ge;2018) that "
            "captures rule changes affecting conversion rates. Features: `down`, `distance`, "
            "`yards_to_goal`, `posteam_total`, `posteam_spread`, `era`. Notably **`era` is the "
            "4th-most-important feature by gain** — the rule-era signal matters."
        ),
        discussion=(
            "Calibration is evaluated by collapsing the 76-class gain distribution into "
            "**P(first down)** (sum of class probabilities for gains &ge; distance-to-go) and "
            "comparing to the empirical first-down rate. On the 2.2M-play corpus the "
            "**first-down calibration MAE is 0.005** — the predicted conversion probabilities "
            "are almost exactly right. The report renders two figures: the first-down "
            "calibration scatter and the feature-importance bar chart (where `era` ranks 4th)."
        ),
        limitations=(
            "The label is `statYardage` — ESPN's recorded yards gained — which can disagree with "
            "official play-by-play on penalty-laden or laterals plays, adding label noise at the "
            "tails of the gain distribution. The model covers gains -10..65; rare plays outside "
            "that window are clipped. It predicts a *yardage* distribution, not the binary "
            "go/no-go decision itself — the decision EV is computed downstream by combining this "
            "distribution with the EP/WP surfaces."
        ),
        features=(
            "**6 features**; one row per scrimmage (3rd/4th-down) play. The label is the integer "
            "yards gained, shifted into 76 ordinal classes (-10..65).\n\n"
            "| Feature | Type | What it encodes |\n"
            "|---|---|---|\n"
            "| `down` | numeric | Current down (3 or 4). |\n"
            "| `distance` | numeric | Yards to go — the conversion threshold. |\n"
            "| `yards_to_goal` | numeric | Field position (compresses the gain distribution near "
            "the goal line). |\n"
            "| `posteam_total` | numeric | Possession-team game total (proxy for offensive "
            "quality / pace). |\n"
            "| `posteam_spread` | numeric | Possession-team spread (game-script context). |\n"
            "| `era` | ordinal | CFB rule era (0:&le;2006, 1:2007-13, 2:2014-17, 3:&ge;2018) — "
            "captures rule changes affecting conversion rates. **Ranks 4th by gain.** |\n"
        ),
        model=(
            "**Algorithm.** XGBoost, `objective=multi:softprob` over **76 classes** (integer "
            "gains -10..65), **157 boosting rounds**. Lineage is the cfb4th yards model plus an "
            "`era` rule-era factor. P(first down) for any distance-to-go is recovered by "
            "summing class probabilities for gains &ge; the distance.\n\n"
            "**Evaluation.** Calibration collapses the 76-class distribution into P(first down) "
            "and compares to the empirical conversion rate over the 2.2M-play corpus (a sampled "
            "subset for the figure). This is a fit/calibration check on the full corpus rather "
            "than a season-held-out LOSO pass.\n\n"
            "**Rule-era variant (adopted).** Switching the ordinal `era` factor to the one-hot "
            "dummies (`era0..era3`) materially improves out-of-fold first-down calibration — "
            "pooled LOSO cal-MAE **0.0035 → 0.0027**. Shipped side-by-side as `fd_model_era.ubj` "
            "(9 features, era one-hot replacing the ordinal factor)."
        ),
        importance=(
            "By XGBoost gain: `distance` leads (it *is* the conversion threshold), then "
            "`yards_to_goal` and the team total/spread context; **`era` ranks 4th** — confirming "
            "the rule-era signal is real, not decorative. The feature-importance bar chart is "
            "rendered alongside the calibration plot."
        ),
    ),
    "rb_eval": ModelNarrative(
        summary=(
            "The RB-evaluation (xREPA) model maps a running back's per-play efficiency to an "
            "**expected** rushing EPA, isolating the back's contribution from team/context. It "
            "is an **analytic artifact** used for player evaluation and is **not bundled into "
            "sdv-py** (it ships only here, with the model program)."
        ),
        recipe=(
            "A `pygam` **LinearGAM(s(0) + s(1))** — two smooth splines — mapping "
            "`epa_per_play` and `success` (rate) to `unadjusted_epa`. Fit on **897 rushers "
            "across 2015-2025**. The smooth, monotone-ish response surface is deliberately "
            "simple and interpretable rather than a high-variance tree ensemble."
        ),
        discussion=(
            "Validation is leave-one-season-out: bin the predicted EPA, compare to mean realized "
            "unadjusted EPA per bin, and report a weighted R² and weighted calibration error "
            "(the `rb_eval.validate` recipe, a port of the R `show_calibration_chart`). The "
            "calibration figure plots binned predicted EPA/play against realized EPA. Metrics "
            "are emitted only when the `xrepa_loso.parquet` is present (regenerate with "
            "`python -m rb_eval train`)."
        ),
        limitations=(
            "With only 897 rushers the sample is small, so season-to-season bins can be thin at "
            "the EPA extremes. The two-feature GAM intentionally ignores offensive-line, "
            "scheme, and opponent adjustments — it estimates the *unadjusted* expected EPA, "
            "leaving those adjustments to downstream layers. Because it is an analytic artifact "
            "(pickled GAM, not an XGBoost booster), it is excluded from the sdv-py model bundle."
        ),
        features=(
            "**2 features**; one row per qualifying rusher-season. The target is the rusher's "
            "`unadjusted_epa`.\n\n"
            "| Feature | Type | What it encodes |\n"
            "|---|---|---|\n"
            "| `epa_per_play` | smooth `s(0)` | The rusher's mean EPA per rush — the primary "
            "efficiency signal. |\n"
            "| `success` | smooth `s(1)` | The rusher's success rate (share of positive-EPA "
            "rushes) — the consistency signal. |\n"
        ),
        model=(
            "**Algorithm.** A `pygam` **`LinearGAM(s(0) + s(1))`** — two penalized smoothing "
            "splines — deliberately simple and interpretable rather than a high-variance tree "
            "ensemble. Fit on **897 rushers across 2015-2025**.\n\n"
            "**Evaluation.** Leave-one-season-out: bin the predicted EPA, compare to mean realized "
            "unadjusted EPA per bin, and report a weighted R² and weighted calibration error (the "
            "`rb_eval.validate` recipe, a port of the R `show_calibration_chart`). Metrics + the "
            "calibration figure are emitted only when `xrepa_loso.parquet` is present (regenerate "
            "with `python -m rb_eval train`)."
        ),
        importance=(
            "Both splines contribute monotone-increasing surfaces: higher `epa_per_play` and "
            "higher `success` both raise expected EPA, with `epa_per_play` carrying the steeper "
            "response. There is no tree-gain importance for a GAM; the spline partial-dependence "
            "shapes are the interpretable analogue."
        ),
    ),
    "fg": ModelNarrative(
        summary=(
            "The field-goal model estimates the probability a placekick is **made**, given only "
            "the kick distance. The single input is `yards_to_goal` (the kick distance is "
            "`yards_to_goal + 17`). It is **bundled in sdv-py** and powers the field-goal branch "
            "of the fourth-down decision surface: the expected value of attempting a field goal is "
            "this make probability times three points."
        ),
        recipe=(
            "A **1-feature** XGBoost **binary:logistic** make-probability model over "
            "**42,589 attempts** (~73% make rate), **60 trees**. Distance is everything — with one "
            "feature the model *is* the empirical make-rate-by-distance curve, smoothed by the "
            "boosting. LOSO weighted calibration error is **0.0085**: binned predicted make "
            "probability equals the empirical make rate to three decimals at every distance."
        ),
        discussion=(
            "Metrics are pooled **leave-one-season-out (LOSO)** out-of-fold predictions over "
            "2004-2025. The headline number is the **weighted calibration error of 0.0085** — the "
            "predicted make probabilities are essentially exact across the whole distance range. "
            "The calibration figure bins the predicted make probability into 0.05 buckets and "
            "plots it against the empirical make rate (point size = n, y=x reference); because the "
            "lone feature is distance, this doubles as the make-prob-vs-distance curve."
        ),
        limitations=(
            "The model is blind to everything except distance — no kicker identity, no weather, no "
            "wind, no surface, no snap/hold quality. Long attempts are thin in the data, so the "
            "fit **extrapolates past ~59 yards** on very few examples and should be read with "
            "caution there. Because it is distance-only, it captures the league-average make curve, "
            "not a particular kicker's leg."
        ),
        features=(
            "**1 feature**; one row per field-goal attempt. The binary label is `fg_made`.\n\n"
            "| Feature | Type | What it encodes |\n"
            "|---|---|---|\n"
            "| `yards_to_goal` | numeric | Field position of the snap; the kick distance is "
            "`yards_to_goal + 17`. The **only** input — distance is everything for a placekick. |\n"
        ),
        model=(
            "**Algorithm.** XGBoost, `objective=binary:logistic`, `eval_metric=logloss`, "
            "**60 boosting rounds**, `max_depth=3`, `eta=0.1`, `subsample=0.8`, "
            "`min_child_weight=30`. The predicted probability is the make probability; the "
            "fourth-down FG expected value is `3 * P(make)`.\n\n"
            "**Evaluation.** Leave-one-season-out over 2004-2025 (42,589 attempts): train on the "
            "other seasons, predict the held-out one, pool the out-of-fold make probabilities. The "
            "pooled weighted calibration error is **0.0085** — predicted equals actual to three "
            "decimals at every distance.\n\n"
            "**Rule-era variant (adopted, modest).** Adding the one-hot era dummies "
            "(`era0..era3`) gives a small but consistent out-of-fold gain — pooled LOSO logloss "
            "**0.5258 → 0.5240**. Shipped side-by-side as `fg_era.ubj` (yards_to_goal + era0..era3)."
        ),
        importance=(
            "There is only one feature, so importance is trivial: `yards_to_goal` carries 100% of "
            "the signal. The interpretable view is the monotone-decreasing make-probability curve "
            "the model traces as distance grows."
        ),
    ),
    "xpass": ModelNarrative(
        summary=(
            "The expected-pass model estimates the probability that a scrimmage play is a "
            "**dropback (pass)** given pre-snap game state — a measure of how *predictable* an "
            "offense's tendency is in a given situation. It is the CFB analogue of the nflfastR "
            "xpass surface, where **`pass_oe = 100 * (pass - xpass)`** is the pass-rate over "
            "expected: positive when an offense passes more than situation-average."
        ),
        recipe=(
            "A **7-feature** XGBoost **binary:logistic** dropback-probability model over "
            "**1.9M scrimmage plays** (1,902,317 rows), **150 trees**. Features are the pre-snap "
            "situation: `down`, `distance`, `yards_to_goal`, `pos_score_diff`, `TimeSecsRem`, "
            "`era`, `period`. **Down dominates** by gain (830) — down/distance is the backbone of "
            "play-calling tendency. LOSO weighted calibration error is **0.0073**."
        ),
        discussion=(
            "Metrics are pooled **leave-one-season-out (LOSO)** out-of-fold predictions over "
            "2004-2025. The pooled **weighted calibration error is 0.0073** — predicted P(pass) "
            "tracks the empirical pass rate tightly across the probability range. The calibration "
            "figure **facets by down** (the xPass analogue of the WP quarter facets), so you can "
            "confirm calibration holds on each down — including the obvious-passing-down tails "
            "where tendency is most lopsided. `pass_oe` (pass minus xpass) is the actionable "
            "residual built on top of this surface."
        ),
        limitations=(
            "xPass is a **pre-snap** quantity: it sees down/distance/score/clock/era but **no "
            "personnel, formation, motion, or no-huddle signal**, so it captures the "
            "situation-explainable part of tendency only. Two offenses in identical game state get "
            "the same xpass; the *team* tendency lives in `pass_oe`, not in xpass itself. `era` "
            "contributes the least (gain 26) — it is a coarse rule-era level shift, not a strong "
            "driver."
        ),
        features=(
            "**7 features**, all pre-snap; one row per scrimmage play. The binary label is "
            "`is_pass` (dropback).\n\n"
            "| Feature | Type | What it encodes |\n"
            "|---|---|---|\n"
            "| `down` | numeric | Current down — **the dominant tendency driver** (gain 830). |\n"
            "| `distance` | numeric | Yards to go — second by gain (592); long-distance ⇒ pass. |\n"
            "| `period` | numeric | Quarter (gain 381); late-game script shifts tendency. |\n"
            "| `pos_score_diff` | numeric | Possession-team score differential (gain 240); trailing "
            "teams pass. |\n"
            "| `TimeSecsRem` | numeric | Seconds remaining in the half (gain 214). |\n"
            "| `yards_to_goal` | numeric | Field position (gain 174); red-zone/own-territory "
            "tendency. |\n"
            "| `era` | ordinal | CFB rule era (gain 26); a coarse level shift in pass rate. |\n"
        ),
        model=(
            "**Algorithm.** XGBoost, `objective=binary:logistic`, `eval_metric=logloss`, "
            "**150 boosting rounds**, `max_depth=5`, `eta=0.1`, `subsample=0.8`, "
            "`colsample_bytree=0.8`, `min_child_weight=20`. The predicted probability is "
            "`xpass`; **`pass_oe = 100 * (pass - xpass)`** is the pass-rate-over-expected "
            "residual.\n\n"
            "**Evaluation.** Leave-one-season-out over 2004-2025 (1.9M plays): train on the other "
            "seasons, predict the held-out one, pool the out-of-fold probabilities. Pooled "
            "weighted calibration error **0.0073**; the calibration figure facets by down."
        ),
        importance=(
            "By XGBoost gain: **`down` (830) ≫ `distance` (592) > `period` (381) > "
            "`pos_score_diff` (240) > `TimeSecsRem` (214) > `yards_to_goal` (174) > `era` (26)**. "
            "Down/distance carries the model — exactly the situational backbone of play-calling "
            "tendency — with score/clock/field-position refining it and the rule-era contributing "
            "a small level shift."
        ),
    ),
    "two_pt": ModelNarrative(
        summary=(
            "The two-point-conversion model estimates the probability a two-point attempt "
            "**succeeds**, given game context. It powers the **go-for-2 vs. extra-point** decision "
            "(`add_2pt_probs`): the model's success probability times two points is compared "
            "against the extra-point expected value, where the XP make rate is the empirical CFB "
            "rate **0.9851**."
        ),
        recipe=(
            "A **4-feature** XGBoost **binary:logistic** success-probability model over "
            "**1,622 attempts**, **40 trees**, ~**48.2% base rate**. Features are game-context "
            "only: `posteam_spread`, `posteam_total`, `pos_score_diff`, `era`. The model is "
            "**near-constant** — its predictions range just **0.39-0.60**, capturing slight "
            "game-context variation around the base rate rather than the flat 0.45 cfb4th uses. "
            "LOSO weighted calibration error is **0.028**."
        ),
        discussion=(
            "Metrics are pooled **leave-one-season-out (LOSO)** out-of-fold predictions. With only "
            "1,622 attempts the surface is deliberately shallow (depth-2 trees), and the "
            "predictions hug the **~48% base rate** (range 0.39-0.60). The pooled weighted "
            "calibration error is **0.028** — looser than the high-volume heads, as expected from "
            "the tiny, noisy sample. The single-panel calibration figure bins predicted vs. actual "
            "success; it is **sparse** because the predictions are near-constant, which is the "
            "honest picture for a 1.6K-attempt target."
        ),
        limitations=(
            "The sample is **tiny** (1,622 attempts), so the model is **near-constant** and cannot "
            "resolve fine context — treat it as a slightly-context-adjusted base rate, not a sharp "
            "per-play estimate. It has **no air-yards, play-call, or personnel** inputs and no "
            "defensive context. The calibration bins are sparse by construction; the decision it "
            "feeds (go-for-2 vs. XP) is therefore driven mostly by the ~48% level against the "
            "0.9851 XP make rate, with only small game-context tilts."
        ),
        features=(
            "**4 features**; one row per two-point attempt. The binary label is "
            "`two_point_success`.\n\n"
            "| Feature | Type | What it encodes |\n"
            "|---|---|---|\n"
            "| `posteam_spread` | numeric | Possession-team pregame spread (team-strength proxy). |\n"
            "| `posteam_total` | numeric | Possession-team game total (offensive-quality proxy). |\n"
            "| `pos_score_diff` | numeric | Possession-team score differential (game-script "
            "context). |\n"
            "| `era` | ordinal | CFB rule era (level shift in conversion rate). |\n"
        ),
        model=(
            "**Algorithm.** XGBoost, `objective=binary:logistic`, `eval_metric=logloss`, "
            "**40 boosting rounds**, `max_depth=2`, `eta=0.05`, `subsample=0.9`, "
            "`min_child_weight=40` — a deliberately shallow fit for a 1,622-row target. Predictions "
            "span just **0.39-0.60** around the **~48.2% base rate**.\n\n"
            "**Evaluation.** Leave-one-season-out, pooled out-of-fold. Weighted calibration error "
            "**0.028**. The single-panel calibration figure is sparse because the model is "
            "near-constant — the faithful picture for a tiny sample. It feeds the go-for-2 vs. XP "
            "decision against the empirical XP make rate **0.9851**."
        ),
        importance=(
            "By XGBoost gain the game-context features (`posteam_total`, `posteam_spread`, "
            "`pos_score_diff`) carry what little structure the 1,622-attempt sample supports, with "
            "`era` a coarse level shift. Because the model is near-constant, no single feature "
            "moves the prediction far from the ~48% base rate."
        ),
    ),
    "pregame_wp": ModelNarrative(
        summary=(
            "The pregame Win Probability model (Track 4, the **Five Factors** surface) forecasts "
            "a matchup's outcome from a single composite team-quality signal. It regresses the "
            "**Five-Factors rating differential** (`5FRDiff` — the gap between the two teams' "
            "composite of efficiency, explosiveness, field position, finishing drives, and "
            "turnovers) onto the realized game point margin (`PtsDiff`), then converts the "
            "predicted margin to a win probability via a Gaussian transform. It is the pregame "
            "analogue of the in-game WP heads: no game state, just team strength."
        ),
        recipe=(
            "An XGBoost **regression** (`5FRDiff` → `PtsDiff`), **10 trees**, fit on **37,774 "
            "team-game box scores (2005-2025)** built from CFBD play/drive data via the "
            "5-factor box-score pipeline. The predicted point margin is mapped to a win "
            "probability with the Gaussian transform **`WP = Phi(pred_PtsDiff / std)`** "
            "(`mu = 0`, `std = 16.46`, stored in the card). A single composite factor explains "
            "**R² 0.535** of margin variance — over half — and the resulting WP carries a LOSO "
            "weighted calibration error of just **0.0115**. It is a Track-4 **analytic artifact** "
            "and is **not bundled into sdv-py**."
        ),
        discussion=(
            "Metrics are pooled **leave-one-season-out (LOSO)** out-of-fold predictions over "
            "2005-2025, so they are honest out-of-sample. The single `5FRDiff` feature recovers "
            "**PtsDiff R² 0.535** — one composite rating explains ~54% of point-margin variance — "
            "and the Gaussian-transformed win probability is well-calibrated: **WP weighted "
            "calibration error 0.0115**, **Brier 0.1698** against a **0.500 win base rate**. The "
            "single-panel calibration figure bins predicted pregame WP into 0.05 buckets and plots "
            "it against the empirical win rate (point size = n, y=x reference). In application the "
            "model is fed a team's *recent-average* 5FR to forecast a future opponent; the fitted "
            "object itself is the same-game `5FRDiff` → `PtsDiff` relationship."
        ),
        limitations=(
            "The model rests on a **single composite feature**, so it cannot resolve matchup "
            "detail beyond the rating gap — it is a strong baseline, not a play-level forecaster. "
            "The fit is **explanatory** (same-game `5FRDiff` vs same-game margin); the *pregame* "
            "use applies a team's **recent-average** 5FR to a future game, which shifts the "
            "distribution the fit never saw, so live forecasts are looser than the in-sample R² "
            "suggests. The 5FR inputs are **CFBD-sourced**, so coverage is **FBS-only**. Because "
            "it is a Track-4 analytic artifact, it ships only here with the model program and is "
            "**not bundled into sdv-py**."
        ),
        features=(
            "**1 feature**; one row per team-game box score. The regression target is `PtsDiff` "
            "(game point margin); the win label `win` is used only for the WP calibration.\n\n"
            "| Feature | Type | What it encodes |\n"
            "|---|---|---|\n"
            "| `5FRDiff` | numeric | The difference between the two teams' **Five-Factors** "
            "composite rating — efficiency, explosiveness, field position, finishing drives, and "
            "turnovers rolled into one number. The **only** input: a single team-quality gap. |\n"
        ),
        model=(
            "**Algorithm.** XGBoost **regression** (squared-error objective), **10 boosting "
            "rounds**, mapping `5FRDiff` → `PtsDiff` (game point margin). The point-margin "
            "prediction is converted to a win probability with the Gaussian transform "
            "**`WP = Phi(pred_PtsDiff / std)`**, `mu = 0`, `std = 16.46` (both stored in the model "
            "card). Trained on **37,774 team-game box scores (2005-2025)**.\n\n"
            "**Evaluation.** Leave-one-season-out over 2005-2025: train on the other seasons, "
            "predict the held-out one, pool the out-of-fold values. The point-margin fit reaches "
            "**R² 0.535**, and the Gaussian-transformed win probability has a **weighted "
            "calibration error of 0.0115** (Brier 0.1698, win base rate 0.500) — the single-panel "
            "calibration figure plots binned predicted pregame WP against the empirical win rate."
        ),
        importance=(
            "There is a single feature, so importance is trivial: `5FRDiff` carries 100% of the "
            "signal. The interpretable view is the near-linear `5FRDiff` → `PtsDiff` response (and "
            "the monotone WP curve it induces through the Gaussian transform) — a ~16.5-point "
            "margin standard deviation means a one-rating-point edge moves the win probability "
            "only modestly, which is why even an R² of 0.54 leaves substantial game-to-game noise."
        ),
    ),
}
