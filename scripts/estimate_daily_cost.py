#!/usr/bin/env python3
"""
Estimate daily spend from ablation sweep runs in output/ablation_sweeps/.

Cost reference: output/ablation_sweeps/cost_estimation_20260417_141623/ablation_cost_summary.json
  - cost_estimation_* folders  → read grand_total_cost_usd directly
  - regular YYYYMMDD_* folders → infer mode from ablation name, multiply by reference cost
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


REFERENCE_COST_FILE = (
    "output/ablation_sweeps/cost_estimation_20260417_141623/ablation_cost_summary.json"
)
RETRIEVAL_REFERENCE_COST_FILE = (
    "output/ablation_sweeps/cost_estimation_20260419_153731/ablation_cost_summary.json"
)


def load_reference_costs(ref_path: Path) -> dict:
    data = json.loads(ref_path.read_text())
    first = next(iter(data["ablations"].values()))
    ref = {}
    for mode, info in first["modes"].items():
        processed = info.get("instances_processed", 1)
        total = info.get("instances_total", processed)
        scale = total / processed
        by_model = {m: d["cost_usd"] * scale for m, d in info.get("by_model", {}).items()}
        ref[mode] = {
            "by_model": by_model,
            "total": sum(by_model.values()),
            "avg_per_model": sum(by_model.values()) / len(by_model) if by_model else 0,
        }
    ref["all_total"] = sum(v["total"] for v in ref.values())
    return ref


def is_retrieval(ablation_name: str) -> bool:
    return "retrieval" in ablation_name.lower()


def cost_for_run(base_ref: dict, retrieval_ref: dict, ablation: str, mode: str, model_names: list) -> float:
    ref = retrieval_ref if is_retrieval(ablation) else base_ref
    if mode not in ref:
        return ref["all_total"]
    mode_ref = ref[mode]
    if not model_names:
        return mode_ref["total"]
    return sum(mode_ref["by_model"].get(m, mode_ref["avg_per_model"]) for m in model_names)


def infer_mode(ablation_name: str) -> str | None:
    name = ablation_name.lower()
    if name.startswith("pairwise") or name.endswith("pairwise"):
        return "pairwise"
    if name.startswith("ranking"):
        return "ranking"
    if name.startswith("pointwise"):
        return "pointwise"
    return None


def extract_date(folder_name: str) -> str | None:
    m = re.search(r"(\d{8})_\d{6}", folder_name)
    return m.group(1) if m else None


def fmt_date(d: str) -> str:
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


def process_sweeps_dir(sweeps_dir: Path, base_ref: dict, retrieval_ref: dict) -> dict:
    """
    Returns:
      daily[date] = {
        "direct_cost": float,       # from cost_estimation_* folders
        "estimated_cost": float,    # from regular run folders
        "unknown_runs": list[str],  # ablation names we couldn't classify
        "skipped_runs": int,        # FAILED runs
      }
    """
    daily = defaultdict(lambda: {
        "direct_cost": 0.0,
        "estimated_cost": 0.0,
        "unknown_runs": [],
        "skipped_runs": 0,
    })

    for entry in sorted(sweeps_dir.iterdir()):
        if not entry.is_dir():
            continue

        name = entry.name
        date = extract_date(name)
        if not date:
            continue

        if name.startswith("cost_estimation_"):
            cost_json = entry / "ablation_cost_summary.json"
            if cost_json.exists():
                data = json.loads(cost_json.read_text())
                daily[date]["direct_cost"] += data.get("grand_total_cost_usd", 0.0)
        else:
            summary_json = entry / "ablation_summary.json"
            if not summary_json.exists():
                continue
            try:
                data = json.loads(summary_json.read_text())
            except json.JSONDecodeError:
                continue
            for run in data.get("runs", []):
                if "FAIL" in run.get("status", "").upper():
                    daily[date]["skipped_runs"] += 1
                    continue
                ablation = run.get("ablation", "")
                mode = infer_mode(ablation) or "unknown"

                # read model names from sweep_summary.json
                models = []
                sweep_path = entry / ablation / "sweep_summary.json"
                if sweep_path.exists():
                    try:
                        sweep = json.loads(sweep_path.read_text())
                        for r in sweep.get("runs", []):
                            if r.get("success"):
                                name = r["name"]
                                prefix = ablation + "-"
                                models.append(name[len(prefix):] if name.startswith(prefix) else name)
                    except (json.JSONDecodeError, KeyError):
                        pass

                daily[date]["estimated_cost"] += cost_for_run(base_ref, retrieval_ref, ablation, mode, models)
                if mode == "unknown":
                    daily[date]["unknown_runs"].append(ablation)

    return daily


def print_report(daily: dict, base_ref: dict, retrieval_ref: dict) -> None:
    print("=" * 65)
    print("Daily Ablation Sweep Cost Estimate")
    print("=" * 65)
    print(f"Base costs (all models):      "
          f"pairwise=${base_ref['pairwise']['total']:.2f}  "
          f"ranking=${base_ref['ranking']['total']:.2f}  "
          f"pointwise=${base_ref['pointwise']['total']:.2f}")
    print(f"Retrieval costs (all models): "
          f"pairwise=${retrieval_ref.get('pairwise', {}).get('total', 0):.2f}  "
          f"pointwise=${retrieval_ref.get('pointwise', {}).get('total', 0):.2f}")
    print()

    grand_total = 0.0
    for date in sorted(daily):
        d = daily[date]
        total = d["direct_cost"] + d["estimated_cost"]
        grand_total += total

        print(f"  {fmt_date(date)}   total=${total:.4f}", end="")
        parts = []
        if d["direct_cost"]:
            parts.append(f"direct=${d['direct_cost']:.4f}")
        if d["estimated_cost"]:
            parts.append(f"estimated=${d['estimated_cost']:.4f}")
        if parts:
            print(f"  ({', '.join(parts)})", end="")
        if d["skipped_runs"]:
            print(f"  [{d['skipped_runs']} failed runs skipped]", end="")
        if d["unknown_runs"]:
            print(f"  [used grand-total for: {', '.join(set(d['unknown_runs']))}]", end="")
        print()

    print()
    print(f"  Grand total: ${grand_total:.4f}")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweeps-dir",
        default="output/ablation_sweeps",
        help="Path to the ablation_sweeps directory (default: output/ablation_sweeps)",
    )
    parser.add_argument(
        "--ref",
        default=REFERENCE_COST_FILE,
        help="Path to reference ablation_cost_summary.json",
    )
    parser.add_argument(
        "--retrieval-ref",
        default=RETRIEVAL_REFERENCE_COST_FILE,
        help="Path to retrieval reference ablation_cost_summary.json",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Filter to a specific date (YYYYMMDD or YYYY-MM-DD). Omit to show all days.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    sweeps_dir = (repo_root / args.sweeps_dir).resolve()
    ref_path = (repo_root / args.ref).resolve()
    retrieval_ref_path = (repo_root / args.retrieval_ref).resolve()

    if not sweeps_dir.is_dir():
        sys.exit(f"ERROR: sweeps dir not found: {sweeps_dir}")
    if not ref_path.is_file():
        sys.exit(f"ERROR: reference cost file not found: {ref_path}")
    if not retrieval_ref_path.is_file():
        sys.exit(f"ERROR: retrieval reference cost file not found: {retrieval_ref_path}")

    base_ref = load_reference_costs(ref_path)
    retrieval_ref = load_reference_costs(retrieval_ref_path)
    daily = process_sweeps_dir(sweeps_dir, base_ref, retrieval_ref)

    if args.date:
        filter_date = args.date.replace("-", "")
        daily = {k: v for k, v in daily.items() if k == filter_date}
        if not daily:
            sys.exit(f"No data found for date {filter_date}")

    print_report(daily, base_ref, retrieval_ref)


if __name__ == "__main__":
    main()
