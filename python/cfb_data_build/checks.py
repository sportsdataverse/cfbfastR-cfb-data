"""Build-time gates for silent no-ops.

WHY THIS EXISTS
---------------
On 2026-08-01 `cfb_team_summaries_weekly` was published with
``adj_off_epa`` correlating **0.9928** with its own unadjusted ``EPAplay_off``.
The opponent adjustment had done nothing: ``cfb_adjusted_epa``'s default ridge
penalty was still the glmnet-scale 325, which under sklearn's
``alpha = lambda * n`` convention crushes every team effect to ~0.

Nothing failed. The build ran green, the columns were present, the values were
plausible, every downstream consumer got numbers. The dataset shipped and sat
live for two days. It was found only because someone measured the output.

That is the defining failure mode of this codebase -- components reporting
success while doing nothing (the same shape as the ridge no-op itself, a
`fill_null(0.0)` that is a no-op on Boolean columns, and a verify pass that
only checked the rows it had targeted). The durable fix is not "remember to
check the lambda"; it is to assert on the OUTPUT, so a transform that fails to
transform cannot pass.
"""

from __future__ import annotations

import numpy as np
import polars as pl

#: Above this, an "adjusted" column is indistinguishable from its raw input.
#: A real CFB opponent adjustment lands near 0.6-0.7 (measured on
#: cfb_ratings_weekly: 0.628); the no-op measured 0.993. 0.95 sits in the
#: empty space between, so it catches the failure without tripping on a season
#: that genuinely had little schedule variance.
NOOP_CORR_THRESHOLD = 0.95

#: (adjusted column, the raw column it is derived from)
ADJUSTMENT_PAIRS = (
    ("adj_off_epa", "EPAplay_off"),
    ("adj_def_epa", "EPAplay_def"),
)


def adjustment_report(df: pl.DataFrame, pairs=ADJUSTMENT_PAIRS) -> dict[str, float]:
    """corr(adjusted, raw) for each pair present. Lower = more adjustment."""
    out: dict[str, float] = {}
    for adj, raw in pairs:
        if adj not in df.columns or raw not in df.columns:
            continue
        d = df.select(adj, raw).drop_nulls()
        if d.height < 50:
            continue
        x, y = d[adj].to_numpy(), d[raw].to_numpy()
        if x.std() == 0 or y.std() == 0:
            out[adj] = float("nan")
            continue
        out[adj] = float(np.corrcoef(x, y)[0, 1])
    return out


def assert_adjustment_is_real(
    df: pl.DataFrame, *, threshold: float = NOOP_CORR_THRESHOLD, label: str = ""
) -> dict[str, float]:
    """Raise if an opponent-adjusted column is ~identical to its raw input.

    Call this on any frame carrying ``adj_*`` columns before writing or
    publishing it.

    Raises:
        ValueError: if any adjusted column correlates above ``threshold`` with
            its raw source, i.e. the adjustment did not happen.
    """
    rep = adjustment_report(df)
    bad = {k: v for k, v in rep.items() if v == v and v > threshold}
    if bad:
        detail = ", ".join(f"{k} vs raw corr={v:.4f}" for k, v in bad.items())
        raise ValueError(
            f"opponent adjustment is a NO-OP{' in ' + label if label else ''}: "
            f"{detail} (threshold {threshold}). The ridge penalty is almost "
            f"certainly on the wrong scale -- sklearn uses alpha = lambda * n, "
            f"so a glmnet-scale lambda (e.g. 325) shrinks every team effect to "
            f"zero. Check sportsdataverse.cfb.cfb_adjusted_epa._RIDGE_LAMBDA "
            f"(expected ~0.035) and that no caller overrides it."
        )
    return rep


#: A team-talent table with no blue-chip variation is not measuring talent.
#: Measured on the 2013-2026 raw store: blue_chip_ratio spans 0.000 (Air Force)
#: to 0.563 (Georgia), sd ~0.13. A degenerate build (empty feed, all-null join)
#: collapses this to 0.
MIN_BLUE_CHIP_SD = 0.02


def assert_talent_is_real(df, *, label: str = "") -> None:
    """Raise if a team-talent table is empty or degenerate.

    THE FAILURE THIS PREVENTS. `cfb_roster_talent` returned ZERO ROWS for every
    season from the day it was written (2026-07-08) -- `_PAGE_SIZE` was 500,
    which exceeds what the 247 RDB serves inside the 3s client timeout, so every
    page raised curl(28) and the loop returned a well-formed empty frame. That
    is indistinguishable from "247 has no recruits" to every caller.

    It then flowed into `cfb_recruiting_projection`'s left join with talent on
    the LEFT, so zero talent rows produced zero output rows, and the only thing
    between an empty feed and a silently talent-free projection was
    `build_recruiting`'s zero-row guard. Nothing measured the talent itself.

    Assert on the OUTPUT: a real talent table has rows, non-null composites, and
    blue-chip ratios that actually vary.
    """
    tag = f" [{label}]" if label else ""
    if df.height == 0:
        raise ValueError(
            f"team talent{tag} is EMPTY -- a recruit class is never empty; this is a fetch failure"
        )
    for col in ("talent_composite", "blue_chip_ratio"):
        if col not in df.columns:
            raise ValueError(
                f"team talent{tag} is missing {col!r}; got {sorted(df.columns)}"
            )
        nulls = df[col].null_count()
        if nulls == df.height:
            raise ValueError(
                f"team talent{tag}: {col!r} is entirely null ({nulls}/{df.height})"
            )
    sd = df["blue_chip_ratio"].drop_nulls().std()
    if sd is None or float(sd) < MIN_BLUE_CHIP_SD:
        raise ValueError(
            f"team talent{tag}: blue_chip_ratio sd={sd} < {MIN_BLUE_CHIP_SD} -- "
            "every team looks alike, so the join or the feed collapsed"
        )


#: Returning production that does not vary across teams is not measuring
#: roster continuity. Measured on real seasons: off_returning sd ~0.22-0.24.
MIN_RETURNING_SD = 0.05


def assert_returning_is_real(df, *, label: str = "") -> None:
    """Raise if a returning-production table is empty or degenerate.

    Same shape of failure as `assert_talent_is_real`, different input. Returning
    production is a ratio built from two joins (season S-1 production, season S
    roster); when either side fails to key, every team comes out at 0.0 or null
    and the table still looks well-formed. The 57.7% name-match that preceded
    the id-keyed join is exactly that failure part-way done.
    """
    tag = f" [{label}]" if label else ""
    if df.height == 0:
        raise ValueError(
            f"returning production{tag} is EMPTY -- a season always has returning players"
        )
    for col in ("off_returning", "overall_returning"):
        if col not in df.columns:
            raise ValueError(
                f"returning production{tag} is missing {col!r}; got {sorted(df.columns)}"
            )
        if df[col].null_count() == df.height:
            raise ValueError(f"returning production{tag}: {col!r} is entirely null")
    sd = df["off_returning"].drop_nulls().std()
    if sd is None or float(sd) < MIN_RETURNING_SD:
        raise ValueError(
            f"returning production{tag}: off_returning sd={sd} < {MIN_RETURNING_SD} -- "
            "every team returns the same share, so a join key collapsed"
        )
    mean = df["off_returning"].drop_nulls().mean()
    if mean is not None and not (0.05 <= float(mean) <= 0.95):
        raise ValueError(
            f"returning production{tag}: off_returning mean={mean:.3f} is outside [0.05, 0.95]; "
            "a real season lands near 0.40-0.60"
        )


def assert_passer_epa_includes_sacks(df, *, label: str = "") -> None:
    """Raise if `cfb_passing` looks like it dropped its negative plays again.

    THE FAILURE THIS PREVENTS (cfbfastR-cfb-data#30). `summarize_passer`
    aggregates plays whose derived passer id survives
    (`completion_player_id ?? incompletion_player_id`) -- which is neither a
    sack nor an interception. The counts were revived separately via a
    name->id map; the EPA was not. So every QB's TEPA carried his completions
    and incompletions and none of his worst outcomes, and `EPAplay` ran ~2.8x a
    play-by-play reconstruction. Dylan Raiola 2025 published 0.4801 EPA/play,
    9th nationally, against 0.128 once his 27 sacks and 5 picks were counted.

    Nothing caught it because every column was individually plausible: `sacked`
    was right, `dropbacks` was right, and TEPA was simply a sum over the wrong
    row set. The check therefore has to compare columns TO EACH OTHER.

    NO LEVEL THRESHOLD IS POSSIBLE HERE -- this was measured, not assumed. The
    first version of this gate rejected any qualified passer above 0.45
    EPA/dropback, on the belief that the post-fix leader sat near 0.35.
    Rebuilding 2004-2025 says otherwise: the real post-fix maximum is Jameis
    Winston's 2013 at **0.7666** over 321 dropbacks, which is HIGHER than the
    0.7641 that the broken 2025 build produced. A legitimate Heisman season and
    the bug occupy the same range, so any cutoff either blocks Winston or never
    fires. The two invariants below are exact instead:

    1. `TEPA` must equal `EPAplay * dropbacks`. Before the fix `EPAplay` was
       the mean EPA over ATTEMPTS (`TEPA / plays`) while `dropbacks` was
       `att + sacked`, so the identity broke for every passer who took a sack.
       It holds for all 10,463 post-fix rows across 22 seasons. This is what
       catches the numerator and denominator spanning different play sets --
       the actual defect in #30.
    2. `sack_epa` and `int_epa` must both be present and clearly negative in
       aggregate. That is what dies if the `(pos_team_id, passer_player_name)`
       join breaks or the EPA column stops being carried into it, which is how
       the values would silently stop reaching the passer.
    """
    tag = f" [{label}]" if label else ""
    if df.height == 0:
        raise ValueError(f"passing{tag} is EMPTY")
    need = {
        "EPAplay",
        "TEPA",
        "sacked",
        "pass_int",
        "dropbacks",
        "sack_epa",
        "int_epa",
    }
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"passing{tag} missing {sorted(missing)}")

    # (1) numerator and denominator must span the same play set
    off = df.with_columns(
        _resid=(pl.col("TEPA") - pl.col("EPAplay") * pl.col("dropbacks")).abs()
        - 1e-6 * (1 + pl.col("TEPA").abs())
    ).filter(pl.col("_resid") > 0)
    if off.height:
        worst = off.sort("_resid", descending=True).head(1).to_dicts()[0]
        name = worst.get("passer_player_name", "?")
        raise ValueError(
            f"passing{tag}: TEPA != EPAplay * dropbacks for {off.height} passer(s); "
            f"worst is {name} with TEPA={worst['TEPA']:.3f} vs "
            f"EPAplay*dropbacks={worst['EPAplay'] * worst['dropbacks']:.3f}. "
            "EPA/play is being computed over a different set of plays than TEPA "
            "sums -- the shape of cfbfastR-cfb-data#30."
        )

    # (2) sack and interception EPA must actually have reached the passer
    for col, flag in (("sack_epa", "sacked"), ("int_epa", "pass_int")):
        # `flag` is in `need`, so no absent-column fallback: falling back to the
        # whole frame would let a frame with no interception count pass on an
        # unrelated negative aggregate.
        rows = df.filter(pl.col(flag) > 0)
        if not rows.height:
            continue
        total = float(rows[col].sum())
        if total >= 0:
            raise ValueError(
                f"passing{tag}: aggregate {col} is {total:.1f}, expected clearly "
                f"negative. {col.split('_')[0]} EPA is not reaching the passer."
            )
