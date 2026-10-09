"""Receiving rates over targets ESPN never attributed (E3).

Percentile audit 2026-10-08. Synthetic frames; expected values are worked by hand.
"""

from __future__ import annotations

import polars as pl
import pytest
from cfb_data_build import team_summaries
from cfb_data_build.team_summaries import build_team_summaries

from tests.cfb_data_build.test_summaries_definition_parity import (
    _GROUPS,
    _IDS,
    TEAMS,
    _complete,
    _dropback,
)

#: per-target receiving rates, plus the dispersion of per-game EPA per target
_PER_TARGET = (
    "EPAplay",
    "success",
    "yardsplay",
    "catchpct",
    "EPAplay_sd",
    "EPAplay_p10",
    "EPAplay_p90",
    "boom_rate",
    "bust_rate",
)


def _incomplete(off: str, receiver: str | None) -> dict:
    named = (
        {"target_player": f"WR {off}", "target_player_id": receiver} if receiver else {}
    )
    return _dropback(
        off,
        pass_attempt=1.0,
        play_type="Pass Incompletion",
        EPA=-0.4,
        incompletion_player=f"QB {off}",
        incompletion_player_id=f"{off}Q",
        **named,
    )


def _receiving_build(
    monkeypatch, *, name_every_receiver: bool
) -> dict[str, pl.DataFrame]:
    # each team: catches (two for "1", three for "2", so the rates rank) and one
    # incompletion to its WR, plus one incompletion that names a receiver only when
    # `name_every_receiver` (unattributed share 0 vs 2/4 = 0.5)
    rows = []
    for off, catches in (("1", 2), ("2", 3)):
        wr = f"{off}W"
        rows += [_complete(off, 10 - 2 * k) for k in range(catches)]
        rows += [
            _incomplete(off, wr),
            _incomplete(off, wr if name_every_receiver else None),
        ]
    plays = pl.DataFrame(
        rows, schema_overrides={c: pl.Utf8 for c in _IDS}, infer_schema_length=None
    ).with_row_index("game_play_number", offset=1)
    home = pl.col("pos_team") == pl.col("home")
    plays = plays.with_columns(
        **{
            f"{side}_EPA_{kind}": pl.when(is_side & (pl.col(kind) == 1)).then(
                pl.col("EPA")
            )
            for side, is_side in (("home", home), ("away", ~home))
            for kind in ("pass", "rush")
        }
    )
    adjusted = pl.DataFrame(
        {"team_id": ["1", "2"], "pos_team": [TEAMS["1"], TEAMS["2"]]}
        | {
            c: [0.1, -0.1]
            for c in ("adj_off_epa", "adj_def_epa", "net_adj_epa")
            + ("off_strength_faced", "def_strength_faced")
        }
    )
    monkeypatch.setattr(team_summaries, "cfb_adjusted_epa", lambda _plays: adjusted)
    # a two-receiver position cohort, so _pos_pct is live when the rates are
    monkeypatch.setattr(team_summaries, "MIN_COHORT_PLAYERS", 1)
    rosters = pl.DataFrame(
        {"athlete_id": ["1W", "2W"], "position_abbreviation": ["WR", "WR"]}
    )
    return build_team_summaries(
        plays, 2025, through_week=1, groups=_GROUPS, rosters=rosters
    )


@pytest.mark.parametrize("named", [True, False], ids=["attributed", "unattributed"])
def test_per_target_receiving_rates_blank_when_most_incompletions_name_no_receiver(
    monkeypatch, named
):
    """E3: above the share, every per-target rate and its rank/percentiles is null.

    The counts stay, and the league baseline takes no mean over the blanked column.
    """
    tables = _receiving_build(monkeypatch, name_every_receiver=named)
    wr = tables["receiving"].sort("player_id").row(0, named=True)
    league = tables["league_averages"].filter(pl.col("category") == "receiving")
    # raw counts are published either way; targets are the attributed ones
    assert (wr["comp"], wr["yards"]) == (2, 18.0)
    assert wr["targets"] == (4 if named else 3)
    assert "targets" in league["metric"].to_list()
    if named:
        assert wr["catchpct"] == pytest.approx(0.5)
        assert wr["catchpct_rank"] is not None and wr["catchpct_pos_pct"] is not None
        assert "catchpct" in league["metric"].to_list()
        return
    for m in _PER_TARGET:
        assert wr[m] is None, m
    for m in ("EPAplay", "success", "yardsplay", "catchpct", "boom_rate"):
        for sfx in ("_rank", "_pct", "_pos_pct"):
            assert wr[f"{m}{sfx}"] is None, f"{m}{sfx}"
    assert not set(_PER_TARGET) & set(league["metric"].to_list())
