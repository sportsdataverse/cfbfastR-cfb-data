"""CLI: build-boxes | train | predict-matchup.

Usage:
  uv run python -m pregame_wp build-boxes  --seasons 2012:2020 --raw-dir cfb/pregame_wp/raw/ --out cfb/pregame_wp/boxes/
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
    bb.add_argument("--raw-dir", default="cfb/pregame_wp/raw/",
                    help="Per-game JSON cache root ({raw_dir}/{game_id}/plays.json).")
    bb.add_argument("--out", default="cfb/pregame_wp/boxes/box-scores.parquet",
                    help="Output parquet path (a directory gets box-scores.parquet appended).")
    bb.add_argument("--no-fetch", action="store_true",
                    help="Do not hit CFBD; build only from already-cached JSON.")
    bb.add_argument("--quiet", action="store_true", help="Suppress per-game progress.")

    tr = sub.add_parser("train", help="Train XGBRegressor on stored game boxes.")
    tr.add_argument("--boxes", default="cfb/pregame_wp/boxes/")
    tr.add_argument("--out", default="cfb/pregame_wp/")

    pm = sub.add_parser("predict-matchup", help="Predict WP for a future matchup.")
    pm.add_argument("--home", required=True)
    pm.add_argument("--away", required=True)
    pm.add_argument("--year", type=int, required=True)
    pm.add_argument("--boxes", default="cfb/pregame_wp/boxes/box-scores.parquet",
                    help="Box parquet (file or directory of *.parquet) for season-strength tables.")
    pm.add_argument("--model", default="cfb/pregame_wp/pgwp_model.ubj")
    pm.add_argument("--games", type=int, default=4,
                    help="Recent games to average for each team's 5FR.")
    pm.add_argument("--week", type=int, default=-1,
                    help="Week of season (-1 = consider all weeks; 0 = preseason / prior year).")
    pm.add_argument("--neutral-site", action="store_true",
                    help="Neutral site: no home-field advantage applied.")
    pm.add_argument("--covid", action="store_true",
                    help="Use reduced COVID-2020 HFA (+1.0 instead of +2.5).")
    pm.add_argument("--no-sos", action="store_true",
                    help="Skip CFBD SoS fetches (conferences + talent); use raw 5FR only.")

    return ap


def _read_boxes(boxes_arg: str):
    """Load the box corpus from a parquet file or a directory of *.parquet."""
    import glob as _glob

    import pandas as pd

    p = Path(boxes_arg)
    if p.is_dir():
        files = sorted(_glob.glob(str(p / "*.parquet")))
        if not files:
            raise FileNotFoundError(f"no parquet files in {p}")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if not p.exists():
        raise FileNotFoundError(f"box parquet not found: {p}")
    return pd.read_parquet(p)


def _parse_seasons(seasons_str: str) -> list[int]:
    parts = seasons_str.split(":")
    if len(parts) == 2:
        return list(range(int(parts[0]), int(parts[1]) + 1))
    return [int(parts[0])]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "build-boxes":
        from pregame_wp.ep_curve import load_ep_curve, load_punt_sr
        from pregame_wp.training import build_training_frame

        seasons = _parse_seasons(args.seasons)
        ep_data = load_ep_curve()
        punt_sr = load_punt_sr()
        stored = build_training_frame(
            seasons,
            raw_dir=args.raw_dir,
            ep_data=ep_data,
            punt_sr=punt_sr,
            fetch_missing=not args.no_fetch,
            verbose=not args.quiet,
        )
        if stored.empty:
            print("build-boxes: no games processed (no cached/fetched data?)")
            return 1

        out = Path(args.out)
        if out.is_dir() or args.out.endswith(("/", "\\")):
            out = out / "box-scores.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        stored.to_parquet(out, index=False)
        print(
            f"build-boxes: wrote {len(stored)} team-game rows "
            f"({stored['GameID'].nunique()} games) → {out}"
        )

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
        print(f"train: model saved → {out_dir / 'pgwp_model.ubj'} (mu={mu}, std={std:.4f})")

    elif args.cmd == "predict-matchup":
        from pregame_wp.predict import predict_matchup
        from pregame_wp.training import load_pgwp_model

        model_path = Path(args.model)
        if not model_path.exists():
            print(f"predict-matchup: model not found at {model_path}")
            return 1
        try:
            stored = _read_boxes(args.boxes)
        except FileNotFoundError as exc:
            print(f"predict-matchup: {exc}")
            return 1

        model, mu, std = load_pgwp_model(str(model_path))

        # --- strength-of-schedule inputs (CFBD; gracefully skipped on failure) ---
        conferences = None
        roster_talent = returning_production = None
        if not args.no_sos:
            from pregame_wp import data_ingest as di
            try:
                conferences = di.build_conferences(di.fetch_teams(args.year))
            except Exception as exc:  # noqa: BLE001
                print(f"predict-matchup: conference SoS skipped ({exc})")
            # roster-talent x returning-production only feed the weeks 1-4 adjustment.
            if 0 < args.week <= 4:
                try:
                    import pandas as pd
                    from pregame_wp.talent import calculate_returning_production, calculate_roster_talent
                    rec = pd.concat(
                        [di.build_recruiting_df(di.fetch_recruiting_teams(y))
                         for y in range(args.year - 3, args.year + 1)],
                        ignore_index=True,
                    )
                    tmap = dict(zip(*calculate_roster_talent(rec, args.year)[["team", "talent"]].values.T))
                    rdf = calculate_returning_production(di.build_returning_df(di.fetch_returning_production(args.year)))
                    rmap = dict(zip(*rdf[["team", "returning_production"]].values.T))
                    roster_talent = lambda team, _yr: float(tmap.get(team, 0.0))  # noqa: E731
                    returning_production = lambda team, _yr: float(rmap.get(team, 0.0))  # noqa: E731
                except Exception as exc:  # noqa: BLE001
                    print(f"predict-matchup: talent/returning SoS skipped ({exc})")

        win_prob, proj_mov = predict_matchup(
            args.home, args.away, args.year,
            week=args.week,
            games_to_consider=args.games,
            stored_game_boxes=stored,
            model=model, mu=mu, std=std,
            conferences=conferences,
            roster_talent=roster_talent,
            returning_production=returning_production,
            adjust_hfa=not args.neutral_site,
            adjust_covid=args.covid,
        )
        favored = args.home if proj_mov >= 0 else args.away
        print(
            f"predict-matchup: {args.away} @ {args.home}, year={args.year}, week={args.week}"
        )
        print(
            f"  proj MOV (home {args.home}): {proj_mov:+.2f} → "
            f"{favored} favored; P({args.home} win) = {win_prob:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
