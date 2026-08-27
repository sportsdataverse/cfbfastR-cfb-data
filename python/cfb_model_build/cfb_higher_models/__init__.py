"""Higher-order CFB models: game-level and season-level heads over the
play-level (EP/WP/adj-EPA) substrate.

Four styles, distinguished by their INFORMATION BOUNDARY -- what data a model
is allowed to see -- and therefore by the metric that can honestly judge them:

    expectancy   same-game plays        "what should this game have produced?"
    explanatory  full season, in-sample "what drove the season?"
    predictive   strictly as-of W-1     "what happens next week?"
    projection   preseason / as-of      "how does the season end?"

These are not four kinds of quality; they are four questions. A model is not
wrong for being explanatory -- it is wrong for being explanatory and labelled
predictive. Every head declares its boundary in its ``Report``, and the
as-of join lives in exactly one function (:func:`data.build_game_frame`).
"""

from .data import build_game_frame, diff_features, feature_columns, paired_features
from .metrics import Report, baselines, evaluate, margin_metrics, norm_cdf, wp_metrics

__all__ = [
    "Report",
    "baselines",
    "build_game_frame",
    "diff_features",
    "evaluate",
    "feature_columns",
    "margin_metrics",
    "norm_cdf",
    "paired_features",
    "wp_metrics",
]
