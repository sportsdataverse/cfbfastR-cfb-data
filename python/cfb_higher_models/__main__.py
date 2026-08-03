"""``python -m cfb_higher_models <cmd>``."""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cfb_higher_models")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("backtest", "fit-pregame", "train-game"):
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
    args = ap.parse_args(argv)

    if args.cmd == "backtest":
        from .backtest import main as run

        return run(args.seasons)
    if args.cmd == "fit-pregame":
        from .fit_pregame import main as run

        return run(args.seasons, args.out_dir)
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
