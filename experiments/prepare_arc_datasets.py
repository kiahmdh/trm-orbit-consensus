import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = Path(
    os.environ.get("TRM_ROOT", PROJECT_ROOT / "external" / "TinyRecursiveModels")
).resolve()
if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from dataset.build_arc_dataset import DataProcessConfig, convert_dataset


@dataclass(frozen=True)
class DatasetRecipe:
    name: str
    subsets: tuple[str, ...]
    test_set_name: str
    output_dir: str


RECIPES = {
    "arc1": DatasetRecipe(
        name="arc1",
        subsets=("training", "evaluation", "concept"),
        test_set_name="evaluation",
        output_dir="data/arc1concept-aug-1000",
    ),
    "arc2": DatasetRecipe(
        name="arc2",
        subsets=("training2", "evaluation2", "concept"),
        test_set_name="evaluation2",
        output_dir="data/arc2concept-aug-1000",
    ),
}


def ensure_new_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty dataset directory: {path}. "
            "Move it aside or choose a different --output-dir."
        )
    path.mkdir(parents=True, exist_ok=True)


def build_augmented(
    prefix: str,
    subsets: list[str],
    test_set_name: str,
    out_dir: str,
    num_aug: int,
    seed: int = 42,
) -> None:
    ensure_new_output_dir(Path(out_dir))
    cfg = DataProcessConfig(
        input_file_prefix=prefix,
        output_dir=out_dir,
        subsets=subsets,
        test_set_name=test_set_name,
        seed=seed,
        num_aug=num_aug,
    )
    convert_dataset(cfg)


def git_revision(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a checkpoint-aligned ARC dataset without mixing ARC-1 and ARC-2 IDs."
    )
    parser.add_argument("--dataset", choices=tuple(RECIPES), default="arc1")
    parser.add_argument("--num-aug", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        help="Override the recipe output directory (required for non-1000 ablations).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_aug < 0:
        raise ValueError("--num-aug must be non-negative")
    if args.num_aug != 1000 and args.output_dir is None:
        raise ValueError("Non-1000 builds require an explicit --output-dir")

    recipe = RECIPES[args.dataset]
    prefix = UPSTREAM_ROOT / "kaggle" / "combined" / "arc-agi"
    output_dir = Path(args.output_dir or recipe.output_dir)
    if not output_dir.is_absolute():
        output_dir = UPSTREAM_ROOT / output_dir

    build_augmented(
        prefix=str(prefix),
        subsets=list(recipe.subsets),
        test_set_name=recipe.test_set_name,
        out_dir=str(output_dir),
        num_aug=args.num_aug,
        seed=args.seed,
    )

    manifest = {
        "recipe": asdict(recipe),
        "output_dir": str(output_dir),
        "input_file_prefix": str(prefix),
        "seed": args.seed,
        "num_aug": args.num_aug,
        "code_revision": git_revision(UPSTREAM_ROOT),
    }
    manifest_path = output_dir / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("Prepared checkpoint-aligned dataset:")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
