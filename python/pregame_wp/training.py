"""Training pipeline: box-score corpus build + outlier filter + XGBRegressor fit.

OQ-7 resolution: mu=0.0 (point-differential is symmetric), std = std of full
training-set predictions.  The notebook used test-split statistics which is
non-reproducible without a fixed seed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

from .box_score import _BAD_TYPES, _ST_TYPES, calculate_box_score_from_frames
from .constants import OUTLIER_Z_5FR, OUTLIER_Z_PTS, WP_MU, XGB_N_ESTIMATORS, XGB_SEED
from .play_features import add_play_features


# ---------------------------------------------------------------------------
# Box-score corpus build (CLI `build-boxes` driver)
# ---------------------------------------------------------------------------

# Box-score column set produced by calculate_box_score_from_frames(), plus the
# four identity / target columns build_training_frame appends per game.
_GAME_META_COLS = ["GameID", "Season", "Week", "Team", "PtsDiff"]


def _game_points(game: dict[str, Any], side: str) -> Optional[float]:
    """Pull a side's final score from a CFBD game record (camel or snake)."""
    for key in (f"{side}Points", f"{side}_points"):
        if key in game and game[key] is not None:
            try:
                return float(game[key])
            except (TypeError, ValueError):
                return None
    return None


def _team_name(game: dict[str, Any], side: str) -> str:
    for key in (f"{side}Team", f"{side}_team"):
        if game.get(key):
            return str(game[key])
    return ""


def _global_eqppp_bounds(
    enriched_plays: list[pd.DataFrame],
) -> tuple[float, float]:
    """Global EqPPP min/max across the training corpus (notebook cell 20).

    Matches ``pbp_data.EqPPP.min()/.max()`` computed over every play that feeds
    the box-score pipeline — these bounds set the explosiveness translate-domain.
    Falls back to the notebook's (-2.0, 2.0) defaults if no EqPPP is available.
    """
    vals: list[float] = []
    for df in enriched_plays:
        if "EqPPP" in df.columns and len(df):
            series = pd.to_numeric(df["EqPPP"], errors="coerce").dropna()
            if len(series):
                vals.append(float(series.min()))
                vals.append(float(series.max()))
    if not vals:
        return -2.0, 2.0
    return min(vals), max(vals)


def build_training_frame(
    seasons: list[int],
    raw_dir: "Path | str",
    ep_data: list[float],
    punt_sr: dict[int, float],
    *,
    games_provider: "Optional[Callable[[int], list[dict[str, Any]]]]" = None,
    frames_loader: "Optional[Callable[[str, Path], tuple[pd.DataFrame, pd.DataFrame]]]" = None,
    fetch_missing: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build the ``stored_game_boxes`` corpus (CLI ``build-boxes`` driver).

    Loops over every game in ``seasons``, computes its per-team Five-Factors box
    score from cached CFBD ``/plays`` + ``/drives`` frames, threads the *real*
    global ``EqPPP`` min/max (computed over the full training PBP, per notebook
    cell 20) into the explosiveness translate-domain, and stacks all team-game
    rows into one frame.

    The four per-game identity / target columns appended to each box score are
    ``GameID``, ``Season``, ``Week``, and ``PtsDiff`` (point differential from
    the game record's final score, antisymmetric across the two teams — exactly
    as the notebook's ``calculate_box_score`` builds them).

    Args:
        seasons: Season years to include (e.g. ``[2018, 2019, 2020]``).
        raw_dir: Root directory of the per-game JSON cache
            (``{raw_dir}/{game_id}/plays.json`` + ``drives.json``).
        ep_data: EP curve list (len 101) from ``ep_curve.load_ep_curve()``.
        punt_sr: ``{yardline: ExpPuntNet}`` from ``ep_curve.load_punt_sr()``.
        games_provider: Override for the per-season games list. Defaults to
            ``data_ingest.fetch_games(season)``. Injectable for offline tests.
        frames_loader: Override for ``(game_id, raw_dir) -> (plays, drives)``.
            Defaults to ``data_ingest.load_game_frames``. Injectable for tests.
        fetch_missing: If True (default), missing per-game JSON is fetched +
            cached via ``data_ingest.fetch_and_cache`` before loading.
        verbose: Print per-game progress.

    Returns:
        A long DataFrame: two rows per processed game (one per team) with the
        full box-score column set plus ``GameID`` / ``Season`` / ``Week`` /
        ``PtsDiff``. Empty (zero-row) frame if no game could be processed.
    """
    from . import data_ingest

    raw_dir = Path(raw_dir)
    _games = games_provider or (lambda yr: data_ingest.fetch_games(season=yr))
    _load = frames_loader or data_ingest.load_game_frames

    # ---- gather per-game inputs (one disk/network load each) ----
    loaded: list[dict[str, Any]] = []
    enriched_plays: list[pd.DataFrame] = []

    for season in seasons:
        try:
            games = _games(season)
        except Exception as exc:  # noqa: BLE001 — surface but continue other seasons
            if verbose:
                print(f"[build-boxes] season {season}: games fetch failed ({exc}); skipping")
            continue

        # /plays + /drives are week-keyed: fetch each week ONCE (not per game) so
        # a full-season build issues ~2 calls/week instead of ~3/game (the per-game
        # path rate-limits CFBD after ~90 games). Each week's plays/drives are then
        # split to all of that week's games and cached per game.
        if fetch_missing:
            from collections import defaultdict
            by_week: dict[int, list] = defaultdict(list)
            for g in games:
                wk = g.get("week")
                if wk is not None:
                    by_week[int(wk)].append(g)
            for wk, wk_games in sorted(by_week.items()):
                if all(
                    (raw_dir / str(g.get("id") or g.get("gameId") or "_") / "plays.json").exists()
                    for g in wk_games
                ):
                    continue
                try:
                    n = data_ingest.fetch_and_cache_week(
                        year=season, week=wk, games=wk_games, raw_dir=raw_dir,
                        season_type=str(
                            wk_games[0].get("seasonType")
                            or wk_games[0].get("season_type") or "regular"
                        ),
                    )
                    if verbose:
                        print(f"[build-boxes] {season} wk{wk}: cached {n} games")
                except Exception as exc:  # noqa: BLE001
                    if verbose:
                        print(f"[build-boxes] {season} wk{wk}: week fetch failed ({exc}); skipping")
                    continue

        # Week order per team is the slot a game occupies in that team's season;
        # the notebook keys opponent_game_ids on a 1-based per-team sequence. The
        # game record's own `week` is the natural, deterministic proxy.
        for i, game in enumerate(games):
            gid = str(game.get("id") or game.get("gameId") or "")
            if not gid:
                continue
            home, away = _team_name(game, "home"), _team_name(game, "away")
            home_pts, away_pts = _game_points(game, "home"), _game_points(game, "away")
            week = game.get("week")

            try:
                plays, drives = _load(gid, raw_dir)
            except FileNotFoundError:
                continue
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"[build-boxes] {gid}: load failed ({exc}); skipping")
                continue

            if plays.empty or drives.empty:
                continue
            if plays["offense"].nunique() != 2:
                # need exactly two teams for calculate_box_score_from_frames
                continue

            # Accumulate EqPPP exactly as the box pipeline sees it (faithful to
            # the explosiveness domain those boxes will be scored against).
            enriched_plays.append(add_play_features(plays, ep_data, _ST_TYPES, _BAD_TYPES))

            pts_diff = None
            if home_pts is not None and away_pts is not None:
                pts_diff = {home: home_pts - away_pts, away: away_pts - home_pts}

            loaded.append({
                "gid": gid, "season": season, "week": week,
                "home": home, "away": away, "pts_diff": pts_diff,
                "plays": plays, "drives": drives,
            })

    if verbose:
        print(f"[build-boxes] loaded {len(loaded)} games; computing global EqPPP bounds...")

    eq_min, eq_max = _global_eqppp_bounds(enriched_plays)
    if verbose:
        print(f"[build-boxes] global EqPPP bounds: min={eq_min:.4f} max={eq_max:.4f}")

    # ---- second pass: build boxes with the real global bounds ----
    box_rows: list[pd.DataFrame] = []
    for g in loaded:
        try:
            box = calculate_box_score_from_frames(
                g["plays"], g["drives"], ep_data, punt_sr,
                eq_ppp_global_min=eq_min, eq_ppp_global_max=eq_max,
            )
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"[build-boxes] {g['gid']}: box-score failed ({exc}); skipping")
            continue

        box["GameID"] = g["gid"]
        box["Season"] = g["season"]
        box["Week"] = g["week"]
        if g["pts_diff"] is not None:
            box["PtsDiff"] = box["Team"].map(g["pts_diff"]).astype(float)
        else:
            box["PtsDiff"] = np.nan
        box_rows.append(box)

    if not box_rows:
        return pd.DataFrame(columns=_GAME_META_COLS)

    stored = pd.concat(box_rows, ignore_index=True)
    if verbose:
        print(f"[build-boxes] built {len(stored)} team-game rows from {len(box_rows)} games")
    return stored


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where 5FRDiff or PtsDiff exceeds the z-score thresholds."""
    mask_5fr = np.abs(stats.zscore(df["5FRDiff"])) < OUTLIER_Z_5FR
    mask_pts = np.abs(stats.zscore(df["PtsDiff"])) < OUTLIER_Z_PTS
    return df[mask_5fr & mask_pts].copy()


def train_pgwp_model(
    df: pd.DataFrame,
) -> tuple[xgb.XGBRegressor, float, float]:
    """Train a 10-tree XGBRegressor on 5FRDiff → PtsDiff.

    Returns:
        model: fitted XGBRegressor
        mu: 0.0 (OQ-7: symmetric by construction)
        std: std of full training-set predictions
    """
    X = df[["5FRDiff"]].values
    y = df["PtsDiff"].values

    model = xgb.XGBRegressor(
        n_estimators=XGB_N_ESTIMATORS,
        seed=XGB_SEED,
        verbosity=0,
    )
    model.fit(X, y)

    preds = model.predict(X)
    mu = WP_MU  # 0.0 — per OQ-7 resolution
    std = float(np.std(preds))

    return model, mu, std


def save_pgwp_model(
    model: xgb.XGBRegressor,
    std: float,
    path: str,
    season_range: tuple[int, int] | None = None,
) -> None:
    """Save model as UBJ + a unified ``model_card.json`` sidecar.

    The card uses the shared ``write_xgb_model_card`` helper (Tracks 1-5 parity)
    and merges the pregame-specific ``mu`` / ``std`` normalization params in at the
    top level via ``extra=``.  Those two keys are load-bearing — both
    ``load_pgwp_model`` and the CLI read them back to reconstruct the
    5FRDiff -> WP Gaussian transform — so the write is intentionally NOT
    best-effort here (a failure must surface, unlike the audit-only cards).
    """
    from pathlib import Path

    from model_training.model_card import write_xgb_model_card

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(p))

    write_xgb_model_card(
        p,
        model_type="pregame_wp",
        label="PtsDiff",
        features=["5FRDiff"],
        model=model,
        seasons=season_range,
        extra={
            "mu": WP_MU,
            "std": std,
            "n_estimators": XGB_N_ESTIMATORS,
            "note": "pgwp_model — NOT bundled into sdv-py. Track 4 analytic artifact.",
        },
    )


def load_pgwp_model(path: str) -> tuple[xgb.XGBRegressor, float, float]:
    """Load model + sidecar metadata (mu, std)."""
    import json
    from pathlib import Path

    p = Path(path)
    model = xgb.XGBRegressor()
    model.load_model(str(p))
    card = json.loads(p.with_suffix(".json").read_text())
    return model, float(card["mu"]), float(card["std"])
