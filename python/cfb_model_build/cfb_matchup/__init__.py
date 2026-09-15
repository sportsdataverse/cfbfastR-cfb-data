"""cfb_matchup -- model family 35: the matchup pipeline's fitted components.

Three artifacts, all plain coefficient / weight tables (no pickled objects):

* ``scoring_opp`` -- logistic glm, P(a drive starting beyond the 40 produces
  a scoring opportunity); applied per drive by
  ``cfb_data_build.matchup_features.build_drive_frame``.
* ``rush_expect`` -- logistic glm, P(rush) from game state; ``rroe`` is
  ``rush - P(rush)`` per play.
* ``wepa_weights`` -- the 120 situational weights of the weighted-EPA
  product (trainer lands with the weight-search port).

Fitting lives in :mod:`cfb_model_build.cfb_matchup.glm`; the CLI in
:mod:`cfb_model_build.cfb_matchup.cli`.
"""

from __future__ import annotations

from cfb_model_build.cfb_matchup.glm import (
    RUSH_EXPECT_ALIASED,
    FitResult,
    fit_logit,
    fit_rush_expect,
    fit_scoring_opp,
    rush_expect_training_frame,
    write_fit,
)

__all__ = [
    "RUSH_EXPECT_ALIASED",
    "FitResult",
    "fit_logit",
    "fit_rush_expect",
    "fit_scoring_opp",
    "rush_expect_training_frame",
    "write_fit",
]
