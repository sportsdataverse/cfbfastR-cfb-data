"""``python -m cfb_higher_models <cmd>``."""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cfb_higher_models")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("backtest", "fit-pregame", "train-game", "experiments"):
        s = sub.add_parser(name)
        s.add_argument("--seasons", nargs="*", type=int)
        if name != "backtest":
            s.add_argument("--out-dir", default="artifacts/higher_models")
        if name == "train-game":
            s.add_argument(
                "--enrich",
                action="store_true",
                help="add prior-season carryover + shrinkage blend features",
            )
            s.add_argument("--blend-k", type=float, default=4.0)
            s.add_argument("--ablate", action="store_true")
        if name == "experiments":
            s.add_argument(
                "--fresh",
                action="store_true",
                help="DROP the parquet cache first. Required after a republish: "
                "the cache is keyed only by season range, so it will serve the "
                "old data to a run that believes it is testing the new.",
            )
            s.add_argument("--quick", action="store_true", help="skip the ablation + k sweep")
    args = ap.parse_args(argv)

    if args.cmd == "backtest":
        from .backtest import main as run

        return run(args.seasons)
    if args.cmd == "fit-pregame":
        from .fit_pregame import main as run

        return run(args.seasons, args.out_dir)
    if args.cmd == "experiments":
        from .experiments import main as run

        return run(args.seasons, args.out_dir, fresh=args.fresh, quick=args.quick)
    from .train_game import main as run

    return run(
        args.seasons,
        args.out_dir,
        enrich=args.enrich,
        blend_k=args.blend_k,
        do_ablate=args.ablate,
    )


if __name__ == "__main__":
    sys.exit(main())
