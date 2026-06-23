"""CLI: build-boxes | train | predict-matchup.

Usage:
  uv run python -m pregame_wp build-boxes  --seasons 2012:2020 --out cfb/pregame_wp/boxes/
  uv run python -m pregame_wp train        --boxes cfb/pregame_wp/boxes/ --out cfb/pregame_wp/
  uv run python -m pregame_wp predict-matchup --home "LSU" --away "Clemson" --year 2019
"""
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="pregame_wp",
        description="CFB Pregame WP + Five-Factors pipeline (Track 4, CFB Modeling Suite).",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    bb = sub.add_parser("build-boxes", help="Compute 5FR box scores from CFBD data.")
    bb.add_argument("--seasons", default="2012:2020",
                    help="Season range as A:B (e.g. 2012:2020).")
    bb.add_argument("--out", default="cfb/pregame_wp/boxes/")
    bb.add_argument("--raw-dir", default=None,
                    help="Per-game CFBD JSON cache dir (default: <out>/raw).")
    bb.add_argument("--season-type", default="regular",
                    choices=["regular", "postseason", "both"])
    bb.add_argument("--limit", type=int, default=0,
                    help="Max games per season to process (0 = all; use a small value to smoke-test).")

    tr = sub.add_parser("train", help="Train XGBRegressor on stored game boxes.")
    tr.add_argument("--boxes", default="cfb/pregame_wp/boxes/")
    tr.add_argument("--out", default="cfb/pregame_wp/")

    pm = sub.add_parser("predict-matchup", help="Predict WP for a future matchup.")
    pm.add_argument("--home", required=True)
    pm.add_argument("--away", required=True)
    pm.add_argument("--year", type=int, required=True)
    pm.add_argument("--model", default="python/pregame_wp/models/pgwp_model.ubj")
    pm.add_argument("--games", type=int, default=4,
                    help="Recent games to average for each team's 5FR.")
    pm.add_argument("--week", type=int, default=-1,
                    help="Week of season (-1 = latest available).")

    return ap


def _parse_seasons(seasons_str: str) -> list[int]:
    parts = seasons_str.split(":")
    if len(parts) == 2:
        return list(range(int(parts[0]), int(parts[1]) + 1))
    return [int(parts[0])]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "build-boxes":
        import pandas as pd

        from pregame_wp.box_score import calculate_box_score_from_frames
        from collections import defaultdict

        from pregame_wp.data_ingest import (
            _norm_team,
            _team_key,
            fetch_games,
            fetch_week_and_cache,
            load_week_game_frames,
        )
        from pregame_wp.ep_curve import load_ep_curve, load_punt_sr

        seasons = _parse_seasons(args.seasons)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = Path(args.raw_dir) if args.raw_dir else out_dir / "raw"
        ep_data = load_ep_curve()
        punt_sr = load_punt_sr()
        print(f"build-boxes: seasons {seasons[0]}–{seasons[-1]}, out={out_dir}, raw={raw_dir}")

        for season in seasons:
            try:
                games = fetch_games(season=season, season_type=args.season_type)
            except Exception as e:  # noqa: BLE001 — a season-level fetch failure shouldn't kill the batch
                print(f"build-boxes {season}: SKIPPED season (games fetch failed: {type(e).__name__}: {e})")
                continue
            if args.limit:
                games = games[: args.limit]
            rows, ok, skipped = [], 0, 0
            # Group this season's games by week, then fetch each week's plays +
            # drives ONCE (two CFBD calls) and slice to games client-side —
            # ~50x fewer API calls than the old per-game fetch.
            by_week: dict[int, list] = defaultdict(list)
            for g in games:
                if g.get("id") is not None and g.get("week") is not None:
                    by_week[int(g["week"])].append(g)
                else:
                    skipped += 1
            for week in sorted(by_week):
                try:
                    fetch_week_and_cache(season, week, raw_dir, season_type=args.season_type)
                except Exception as e:  # noqa: BLE001 — a week fetch failure skips the week, not the batch
                    print(f"  skip week {season} wk{week} (fetch failed: {type(e).__name__}: {e})")
                    skipped += len(by_week[week])
                    continue
                for g in by_week[week]:
                    gid = g.get("id")
                    try:
                        plays_df, drives_df = load_week_game_frames(
                            season, week, raw_dir,
                            _team_key(g, "home"), _team_key(g, "away"),
                            season_type=args.season_type,
                        )
                        box = calculate_box_score_from_frames(plays_df, drives_df, ep_data, punt_sr)
                        box.insert(0, "game_id", gid)
                        box.insert(0, "week", int(week))
                        box.insert(0, "season", season)
                        # per-team actual point differential — the pregame-WP train target
                        home_pts = g.get("homePoints", g.get("home_points"))
                        away_pts = g.get("awayPoints", g.get("away_points"))
                        if home_pts is not None and away_pts is not None:
                            diff_map = {
                                _norm_team(_team_key(g, "home")): float(home_pts) - float(away_pts),
                                _norm_team(_team_key(g, "away")): float(away_pts) - float(home_pts),
                            }
                            box["PtsDiff"] = box["Team"].map(lambda t: diff_map.get(_norm_team(t)))
                        else:
                            box["PtsDiff"] = pd.NA
                        rows.append(box)
                        ok += 1
                    except Exception as e:  # noqa: BLE001 — skip unbuildable games (missing/partial CFBD data)
                        skipped += 1
                        print(f"  skip game {gid} ({season} wk {week}): {type(e).__name__}: {e}")
            if rows:
                season_df = pd.concat(rows, ignore_index=True)
                out_path = out_dir / f"boxes_{season}.parquet"
                season_df.to_parquet(out_path, index=False)
                print(f"build-boxes {season}: {ok} games -> {out_path} "
                      f"({len(season_df)} rows); {skipped} skipped")
            else:
                print(f"build-boxes {season}: 0 games built ({skipped} skipped)")

    elif args.cmd == "train":
        import glob as _glob
        import pandas as pd
        from pregame_wp.training import filter_outliers, save_pgwp_model, train_pgwp_model

        box_files = sorted(_glob.glob(str(Path(args.boxes) / "*.parquet")))
        if not box_files:
            print(f"train: no parquet files found in {args.boxes}")
            return 1
        frames = [pd.read_parquet(f) for f in box_files]
        stored = pd.concat(frames, ignore_index=True)
        filtered = filter_outliers(stored)
        model, mu, std = train_pgwp_model(filtered)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_pgwp_model(model, std, out_dir / "pgwp_model.ubj",
                        season_range=None)
        print(f"train: model saved -> {out_dir / 'pgwp_model.ubj'} (mu={mu}, std={std:.4f})")

    elif args.cmd == "predict-matchup":
        import xgboost as xgb
        import json
        from pregame_wp.predict import five_fr_to_wp

        model_path = Path(args.model)
        if not model_path.exists():
            print(f"predict-matchup: model not found at {model_path}")
            return 1
        m = xgb.XGBRegressor()
        m.load_model(str(model_path))
        card = json.loads(model_path.with_suffix(".json").read_text())
        mu, std = float(card["mu"]), float(card["std"])
        print(f"predict-matchup: {args.home} vs {args.away}, year={args.year}")
        print("  (requires pre-computed 5FR averages; see box_score.py)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
