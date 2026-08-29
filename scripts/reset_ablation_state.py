#!/usr/bin/env python3
"""
reset_ablation_state.py — Reset a batch ablation state file for a clean re-poll.

Clears report_generated / sweep_report_generated flags, deletes stale
scores.json, comparisons/, and cost_report.* files so that the next --poll
re-downloads batch results and regenerates all artifacts from scratch.

Usage:
    python scripts/reset_ablation_state.py <state_file> [<state_file> ...]

Examples:
    python scripts/reset_ablation_state.py output/ablation_sweeps/20260430_094157/ablation_batch_state.json
    python scripts/reset_ablation_state.py output/ablation_sweeps/*/ablation_batch_state.json
"""
import argparse
import json
import pathlib
import shutil
import sys


def reset_state(state_path: pathlib.Path) -> None:
    print(f"\n=== {state_path} ===")
    with open(state_path) as f:
        state = json.load(f)

    for abl_name, entry in state["ablations"].items():
        entry.pop("sweep_report_generated", None)

        for run in entry.get("runs", []):
            run.pop("report_generated", None)

            art_dirs = set()
            for batch in run.get("batches", []):
                run_dir = pathlib.Path(batch["run_dir"])
                art_dirs.add(run_dir.parent)

                scores = run_dir / "scores.json"
                if scores.exists():
                    scores.unlink()
                    print(f"  deleted {scores}")

                cmp_dir = run_dir / "comparisons"
                if cmp_dir.exists():
                    shutil.rmtree(cmp_dir)
                    print(f"  cleared {cmp_dir}")

            for art_dir in art_dirs:
                for name in ("cost_report.json", "cost_report.md"):
                    p = art_dir / name
                    if p.exists():
                        p.unlink()
                        print(f"  deleted {p}")

    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
    print(f"  state reset — ready to re-poll")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("state_files", nargs="+", metavar="STATE_FILE")
    args = parser.parse_args()

    for raw in args.state_files:
        p = pathlib.Path(raw)
        if not p.exists():
            print(f"WARNING: {p} not found — skipping", file=sys.stderr)
            continue
        reset_state(p)


if __name__ == "__main__":
    main()
