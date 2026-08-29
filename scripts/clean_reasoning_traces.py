"""
Copy a reasoning-pairwise-baseline directory with blocklisted comparison files omitted.

The original directory is left untouched. A filtered copy is written to --output-dir
(defaults to <traces-dir>_filtered next to the source).

Uses the same blocklist + index-derivation logic as merge_ablation_runs.py.

Usage:
    python scripts/clean_reasoning_traces.py \\
        --traces-dir output/ablation_sweeps/.../baseline_reasoning/reasoning-pairwise-baseline \\
        --instances output/iclr_test_instances/pairwise_data/.../manipulated/iclr_test_instances.yaml \\
        [--output-dir PATH] \\
        [--dry-run]
"""
import argparse
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from novelty_eval.analysis.filtering import (
    _derive_exclude_indices,
    _load_paper_blocklist,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces-dir", required=True, type=Path, help="Path to reasoning-pairwise-baseline dir")
    parser.add_argument("--instances", required=True, type=Path, help="Path to iclr_test_instances.yaml")
    parser.add_argument("--output-dir", type=Path, default=None, help="Destination for filtered copy (default: <traces-dir>_filtered)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be copied/skipped without writing anything")
    args = parser.parse_args()

    traces_dir: Path = args.traces_dir.resolve()
    instances_path: Path = args.instances.resolve()
    output_dir: Path = (args.output_dir or traces_dir.parent / (traces_dir.name + "_filtered")).resolve()

    if not traces_dir.is_dir():
        sys.exit(f"Error: --traces-dir not found: {traces_dir}")
    if not instances_path.is_file():
        sys.exit(f"Error: --instances not found: {instances_path}")
    if output_dir.exists() and not args.dry_run:
        sys.exit(f"Error: output dir already exists: {output_dir}\nDelete it first or choose a different --output-dir.")

    blocked_titles = _load_paper_blocklist()
    if not blocked_titles:
        print("Blocklist is empty — nothing to filter.")
        return

    print(f"Loaded {len(blocked_titles)} blocked title(s).")

    exclude = _derive_exclude_indices(str(instances_path), blocked_titles)
    if not exclude:
        print("No test instance indices matched the blocklist — nothing to filter.")
        return

    print(f"\nBlocked indices ({len(exclude)}):")
    for idx, title in sorted(exclude.items(), key=lambda x: int(x[0])):
        print(f"  [{idx}] {title}")

    model_dirs = sorted(d for d in traces_dir.iterdir() if d.is_dir())
    if not model_dirs:
        print("\nNo model subdirectories found.")
        return

    blocked_files: set[Path] = set()
    for model_dir in model_dirs:
        comparisons_dir = model_dir / "comparisons"
        if not comparisons_dir.is_dir():
            continue
        for idx in exclude:
            f = comparisons_dir / f"{idx}.txt"
            if f.exists():
                blocked_files.add(f)

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Output: {output_dir}")
    print(f"Copying all files, skipping {len(blocked_files)} blocked comparison file(s).\n")

    skipped = []
    copied = []
    for src in sorted(traces_dir.rglob("*")):
        if src.is_dir():
            continue
        dest = output_dir / src.relative_to(traces_dir)
        if src in blocked_files:
            skipped.append(src.relative_to(traces_dir))
        else:
            copied.append((src, dest))

    print(f"Skipping ({len(skipped)}):")
    for rel in skipped:
        print(f"  {rel}")

    instances_yaml_in_output = output_dir / "iclr_test_instances.yaml"
    instances_yaml_src = next(
        (src for src, _ in copied if src.name == "iclr_test_instances.yaml"),
        None,
    )

    if not args.dry_run:
        for src, dest in copied:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

        if instances_yaml_in_output.exists():
            try:
                from yaml import CSafeLoader as SafeLoader, CSafeDumper as SafeDumper
            except ImportError:
                from yaml import SafeLoader, SafeDumper
            import yaml as _yaml

            with open(instances_yaml_in_output, encoding="utf-8") as f:
                data = _yaml.load(f, Loader=SafeLoader) or {}

            blocked_keys = set(exclude.keys())
            before = len(data)
            data = {k: v for k, v in data.items() if str(k) not in blocked_keys}
            after = len(data)

            with open(instances_yaml_in_output, "w", encoding="utf-8") as f:
                _yaml.dump(data, f, Dumper=SafeDumper, allow_unicode=True, sort_keys=True)

            print(f"\nFiltered iclr_test_instances.yaml: {before} → {after} entries ({before - after} removed).")

        print(f"\nDone. Filtered copy written to:\n  {output_dir}")
    else:
        if instances_yaml_src:
            print(f"\nWould also strip {len(exclude)} blocked entries from iclr_test_instances.yaml in output dir.")
        print(f"\nRe-run without --dry-run to write the copy.")


if __name__ == "__main__":
    main()
