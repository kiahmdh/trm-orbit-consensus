from __future__ import annotations

import argparse
import json
from pathlib import Path

from orbit_consensus.arc1_analysis import run_arc1_normal_analysis


def main() -> int:
    parser = argparse.ArgumentParser(description="Run immutable ARC-v1 normal-cache CPU analysis")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_arc1_normal_analysis(
        cache_dir=args.cache,
        config_path=args.config,
        output_dir=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
