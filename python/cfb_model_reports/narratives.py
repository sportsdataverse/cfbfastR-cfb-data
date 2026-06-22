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


NARRATIVES: dict[str, ModelNarrative] = {
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
            "No LOSO OOF parquet is shipped for the naive variant, so calibration here is "
            "computed on a full-history prediction pass over `pbp_full.parquet` (the naive "
            "feature matrix is deterministic from game state). The naive WP **correlates ~0.94 "
            "with the spread WP** and, as expected, the two **diverge most in the first "
            "quarter** — when the pregame spread carries the most information and the game state "
            "carries the least. By the fourth quarter the mean absolute gap between naive and "
            "spread WP collapses (Q1 ~0.13 vs Q4 ~0.02), because `spread_time` has decayed away "
            "and both models are reading the same near-final game state."
        ),
        limitations=(
            "Because it ignores the market, the naive model is *less sharp* early in games: its "
            "log-loss and Brier are worse than the spread model's (it has strictly less "
            "information). It is the correct tool only when you want a spread-free WP or lack a "
            "spread; for forecasting accuracy when a spread exists, prefer the spread model. The "
            "calibration here is in-sample (resubstitution) rather than LOSO, so read it as a "
            "fit check rather than an out-of-sample guarantee."
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
    ),
}
