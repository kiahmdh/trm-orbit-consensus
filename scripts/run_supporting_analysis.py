from __future__ import annotations

import argparse
import json
from pathlib import Path

from orbit_consensus.supporting_analysis import (
    run_arc1_blank_analysis,
    run_arc2_supporting_analysis,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen CPU-only supporting analyses")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    blank = subparsers.add_parser("arc1-blank")
    blank.add_argument("--normal-cache", type=Path, required=True)
    blank.add_argument("--blank-cache", type=Path, required=True)
    blank.add_argument("--main-results", type=Path, required=True)
    blank.add_argument("--output", type=Path, required=True)

    arc2 = subparsers.add_parser("arc2")
    arc2.add_argument("--cache", type=Path, required=True)
    arc2.add_argument("--main-results", type=Path, required=True)
    arc2.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.mode == "arc1-blank":
        report = run_arc1_blank_analysis(
            normal_cache=args.normal_cache,
            blank_cache=args.blank_cache,
            main_results=args.main_results,
            output_dir=args.output,
        )
    else:
        report = run_arc2_supporting_analysis(
            cache_dir=args.cache,
            main_results=args.main_results,
            output_dir=args.output,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
