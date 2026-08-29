#!/usr/bin/env python3
"""
Generate a Markdown cost report for all ablation sweep runs.

Groups runs by evaluation mode (pointwise / pairwise / ranking / unknown),
shows models run, example counts, estimated cost, and Anthropic vs OpenAI breakdown.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


REFERENCE_COST_FILE = (
    "output/ablation_sweeps/cost_estimation_20260417_141623/ablation_cost_summary.json"
)
RETRIEVAL_REFERENCE_COST_FILE = (
    "output/ablation_sweeps/cost_estimation_20260419_153731/ablation_cost_summary.json"
)

ANTHROPIC_PREFIX = "claude"
OPENAI_PREFIX = "gpt"

MODE_ORDER = ["pointwise", "pairwise", "ranking", "unknown"]


# ── data model ────────────────────────────────────────────────────────────────

@dataclass
class ModelRun:
    name: str
    examples: int | None
    success: bool


@dataclass
class RunEntry:
    folder: str
    timestamp: str
    ablation: str
    description: str
    mode: str
    status: str
    models: list[ModelRun] = field(default_factory=list)
    cost_usd: float = 0.0
    cost_source: str = "estimated"   # "actual" | "estimated" | "grand-total"
    read_error: str | None = None


# ── helpers ───────────────────────────────────────────────────────────────────

def load_reference_costs(ref_path: Path) -> dict:
    """
    Returns nested structure:
      ref["pairwise"]["by_model"]["gpt-5.1"] = scaled full-dataset cost
      ref["pairwise"]["total"]               = sum across all models
      ref["all_total"]                       = grand total across all modes (fallback)
    """
    data = json.loads(ref_path.read_text())
    first = next(iter(data["ablations"].values()))
    ref = {}
    for mode, info in first["modes"].items():
        processed = info.get("instances_processed", 1)
        total = info.get("instances_total", processed)
        scale = total / processed
        by_model = {
            model: mdata["cost_usd"] * scale
            for model, mdata in info.get("by_model", {}).items()
        }
        ref[mode] = {
            "by_model": by_model,
            "total": sum(by_model.values()),
            "avg_per_model": sum(by_model.values()) / len(by_model) if by_model else 0,
        }
    ref["all_total"] = sum(v["total"] for v in ref.values())
    return ref


def is_retrieval(ablation_name: str) -> bool:
    return "retrieval" in ablation_name.lower()


def cost_for_run(
    base_ref: dict,
    retrieval_ref: dict,
    ablation: str,
    mode: str,
    model_names: list[str],
) -> float:
    """Pick the right reference based on ablation name, then sum per-model costs."""
    ref = retrieval_ref if is_retrieval(ablation) else base_ref
    if mode == "unknown" or mode not in ref:
        return ref["all_total"]
    mode_ref = ref[mode]
    if not model_names:
        return mode_ref["total"]
    return sum(mode_ref["by_model"].get(m, mode_ref["avg_per_model"]) for m in model_names)


def infer_mode(name: str) -> str:
    n = name.lower()
    if n.startswith("pairwise") or n.endswith("pairwise"):
        return "pairwise"
    if n.startswith("ranking"):
        return "ranking"
    if n.startswith("pointwise"):
        return "pointwise"
    return "unknown"


def provider(model_name: str) -> str:
    if model_name.startswith(ANTHROPIC_PREFIX):
        return "Anthropic"
    if model_name.startswith(OPENAI_PREFIX):
        return "OpenAI"
    return "Other"


def read_examples(artifact_dir: Path) -> int | None:
    report = artifact_dir / "accuracy_report.txt"
    if not report.exists():
        return None
    m = re.search(r"Number of Test Instances Processed:\s*(\d+)", report.read_text())
    return int(m.group(1)) if m else None


def extract_model_from_run_name(run_name: str, ablation: str) -> str:
    prefix = ablation + "-"
    if run_name.startswith(prefix):
        return run_name[len(prefix):]
    return run_name


# ── parsing ───────────────────────────────────────────────────────────────────

def parse_regular_folder(folder: Path, base_ref: dict, retrieval_ref: dict) -> list[RunEntry]:
    summary_path = folder / "ablation_summary.json"
    if not summary_path.exists():
        return []

    try:
        data = json.loads(summary_path.read_text())
    except json.JSONDecodeError as exc:
        entry = RunEntry(
            folder=folder.name,
            timestamp=folder.name,
            ablation="?",
            description="",
            mode="unknown",
            status="parse-error",
            read_error=f"ablation_summary.json: {exc}",
        )
        return [entry]

    entries = []
    for run in data.get("runs", []):
        ablation = run.get("ablation", "unknown")
        description = run.get("description", "")
        status = run.get("status", "")
        mode = infer_mode(ablation)

        entry = RunEntry(
            folder=folder.name,
            timestamp=folder.name,
            ablation=ablation,
            description=description,
            mode=mode,
            status=status,
        )

        if "FAIL" in status.upper():
            entries.append(entry)
            continue

        # Read sweep_summary for model list
        sweep_summary_path = folder / ablation / "sweep_summary.json"
        if sweep_summary_path.exists():
            try:
                sweep = json.loads(sweep_summary_path.read_text())
                for model_run in sweep.get("runs", []):
                    model_name = extract_model_from_run_name(model_run["name"], ablation)
                    artifact_dir = Path(model_run.get("artifact_dir", ""))
                    examples = read_examples(artifact_dir) if artifact_dir.exists() else None
                    entry.models.append(ModelRun(
                        name=model_name,
                        examples=examples,
                        success=model_run.get("success", False),
                    ))
            except (json.JSONDecodeError, KeyError) as exc:
                entry.read_error = f"sweep_summary.json: {exc}"
        else:
            # Fall back to artifact_dirs in ablation_summary
            for artifact_dir_str in run.get("artifact_dirs", []):
                artifact_dir = Path(artifact_dir_str)
                # Infer model name from path: .../ablation-model/accuracy_test_artifacts/...
                parts = artifact_dir.parts
                try:
                    model_dir_name = parts[parts.index("accuracy_test_artifacts") - 1]
                    model_name = extract_model_from_run_name(model_dir_name, ablation)
                except (ValueError, IndexError):
                    model_name = "unknown"
                examples = read_examples(artifact_dir) if artifact_dir.exists() else None
                entry.models.append(ModelRun(name=model_name, examples=examples, success=True))

        successful_models = [m.name for m in entry.models if m.success]
        entry.cost_usd = cost_for_run(base_ref, retrieval_ref, ablation, mode, successful_models)
        entry.cost_source = "estimated" if mode != "unknown" else "grand-total"
        entries.append(entry)

    return entries


def parse_cost_estimation_folder(folder: Path) -> list[RunEntry]:
    cost_json_path = folder / "ablation_cost_summary.json"
    if not cost_json_path.exists():
        return []

    try:
        data = json.loads(cost_json_path.read_text())
    except json.JSONDecodeError as exc:
        entry = RunEntry(
            folder=folder.name,
            timestamp=folder.name,
            ablation="?",
            description="",
            mode="unknown",
            status="parse-error",
            read_error=f"ablation_cost_summary.json: {exc}",
        )
        return [entry]

    entries = []
    for ablation, abl_data in data.get("ablations", {}).items():
        for mode, mode_data in abl_data.get("modes", {}).items():
            entry = RunEntry(
                folder=folder.name,
                timestamp=folder.name,
                ablation=f"{ablation} ({mode})",
                description=abl_data.get("description", ""),
                mode=mode,
                status="OK",
                cost_usd=mode_data.get("total_cost_usd", 0.0),
                cost_source="actual",
            )
            instances_processed = mode_data.get("instances_processed")
            for model_name, model_data in mode_data.get("by_model", {}).items():
                entry.models.append(ModelRun(
                    name=model_name,
                    examples=instances_processed,
                    success=True,
                ))
            entries.append(entry)

    return entries


def collect_all_entries(
    sweeps_dir: Path,
    base_ref: dict,
    retrieval_ref: dict,
    date_filter: str | None = None,
) -> tuple[list[RunEntry], list[RunEntry]]:
    ok_entries: list[RunEntry] = []
    error_entries: list[RunEntry] = []

    for folder in sorted(sweeps_dir.iterdir()):
        if not folder.is_dir():
            continue
        m = re.search(r"(\d{8})_\d{6}", folder.name)
        if not m:
            continue
        if date_filter and m.group(1) != date_filter:
            continue

        if folder.name.startswith("cost_estimation_"):
            entries = parse_cost_estimation_folder(folder)
        else:
            entries = parse_regular_folder(folder, base_ref, retrieval_ref)

        for e in entries:
            if e.read_error or e.status == "parse-error":
                error_entries.append(e)
            else:
                ok_entries.append(e)

    return ok_entries, error_entries


# ── report generation ─────────────────────────────────────────────────────────

def _cost_badge(source: str) -> str:
    return {"actual": "", "estimated": " *(est.)*", "grand-total": " *(est. full)*"}.get(source, "")


def _model_list(models: list[ModelRun]) -> str:
    if not models:
        return "—"
    parts = []
    for m in models:
        tag = "" if m.success else " ✗"
        parts.append(f"`{m.name}`{tag}")
    return ", ".join(parts)


def _examples(models: list[ModelRun]) -> str:
    counts = {m.examples for m in models if m.examples is not None}
    if not counts:
        return "—"
    return " / ".join(str(c) for c in sorted(counts))


def generate_report(
    ok_entries: list[RunEntry],
    error_entries: list[RunEntry],
    ref_costs: dict[str, float],
    date_filter: str | None = None,
) -> str:
    lines: list[str] = []

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    date_label = f"{date_filter[:4]}-{date_filter[4:6]}-{date_filter[6:]}" if date_filter else "all dates"
    lines += [
        "# Ablation Sweep Cost Report",
        f"*Generated {now} — {date_label}*",
        "",
    ]

    # ── global summary ────────────────────────────────────────────────────────
    total_cost = sum(e.cost_usd for e in ok_entries if "FAIL" not in e.status.upper())
    successful = [e for e in ok_entries if "FAIL" not in e.status.upper() and e.status != "parse-error"]
    failed = [e for e in ok_entries if "FAIL" in e.status.upper()]

    lines += [
        "## Summary",
        "",
        f"| | |",
        f"|---|---|",
        f"| Total estimated cost | **${total_cost:.4f}** |",
        f"| Successful runs | {len(successful)} |",
        f"| Failed runs | {len(failed)} |",
        f"| Read errors | {len(error_entries)} |",
        "",
    ]

    # per-mode summary
    lines += ["### By Mode", ""]
    lines += ["| Mode | Runs | Est. Cost |", "|---|---|---|"]
    for mode in MODE_ORDER:
        mode_entries = [e for e in successful if e.mode == mode]
        mode_cost = sum(e.cost_usd for e in mode_entries)
        lines.append(f"| {mode} | {len(mode_entries)} | ${mode_cost:.4f} |")
    lines.append("")

    # ── Anthropic vs OpenAI ───────────────────────────────────────────────────
    lines += [
        "## Provider Breakdown (Anthropic vs OpenAI)",
        "",
        "Cost is attributed equally across all models in a run.",
        "",
    ]

    provider_cost: dict[str, float] = defaultdict(float)
    provider_runs: dict[str, int] = defaultdict(int)

    for e in successful:
        if not e.models:
            provider_cost["Other"] += e.cost_usd
            provider_runs["Other"] += 1
            continue
        per_model = e.cost_usd / len(e.models)
        for m in e.models:
            p = provider(m.name)
            provider_cost[p] += per_model
            provider_runs[p] += 1

    lines += ["| Provider | Model Runs | Est. Cost |", "|---|---|---|"]
    for p in ["Anthropic", "OpenAI", "Other"]:
        if provider_runs[p]:
            lines.append(f"| {p} | {provider_runs[p]} | ${provider_cost[p]:.4f} |")
    lines.append("")

    # ── per-mode sections ─────────────────────────────────────────────────────
    by_mode: dict[str, list[RunEntry]] = defaultdict(list)
    for e in ok_entries:
        by_mode[e.mode].append(e)

    for mode in MODE_ORDER:
        entries = by_mode.get(mode, [])
        if not entries:
            continue

        lines += [f"## {mode.capitalize()}", ""]

        # group by ablation name for a cleaner table
        by_ablation: dict[str, list[RunEntry]] = defaultdict(list)
        for e in entries:
            by_ablation[e.ablation].append(e)

        lines += [
            "| Folder | Ablation | Status | Models | Examples | Cost |",
            "|---|---|---|---|---|---|",
        ]
        mode_successful = [e for e in entries if "FAIL" not in e.status.upper() and e.status != "parse-error"]
        for ablation in sorted(by_ablation):
            for e in sorted(by_ablation[ablation], key=lambda x: x.folder):
                status_icon = "✅" if e.status == "OK" else ("⚠️" if "parse" in e.status else "❌")
                models_str = _model_list(e.models)
                examples_str = _examples(e.models)
                cost_str = f"${e.cost_usd:.4f}{_cost_badge(e.cost_source)}"
                desc = f" *{e.description}*" if e.description else ""
                lines.append(
                    f"| `{e.folder}` | **{e.ablation}**{desc} "
                    f"| {status_icon} {e.status} | {models_str} | {examples_str} | {cost_str} |"
                )

        # per-provider total for this mode
        mode_provider_cost: dict[str, float] = defaultdict(float)
        for e in mode_successful:
            if not e.models:
                mode_provider_cost["Other"] += e.cost_usd
                continue
            per_model = e.cost_usd / len(e.models)
            for m in e.models:
                mode_provider_cost[provider(m.name)] += per_model
        if mode_provider_cost:
            provider_parts = "  |  ".join(
                f"{p}: **\\${mode_provider_cost[p]:.4f}**"
                for p in ["Anthropic", "OpenAI", "Other"]
                if mode_provider_cost.get(p)
            )
            mode_total = sum(e.cost_usd for e in mode_successful)
            lines += [
                "",
                f"*{mode.capitalize()} total: **\\${mode_total:.4f}** — {provider_parts}*",
            ]

        lines.append("")

    # ── read errors ───────────────────────────────────────────────────────────
    if error_entries:
        lines += ["## Read Errors", ""]
        lines += ["| Folder | Ablation | Error |", "|---|---|---|"]
        for e in error_entries:
            err = e.read_error or e.status
            lines.append(f"| `{e.folder}` | {e.ablation} | {err} |")
        lines.append("")

    # ── reference costs used ──────────────────────────────────────────────────
    lines += [
        "---",
        "## Reference Costs Used (full-dataset estimates)",
        "",
        "*(from `cost_estimation_20260417_141623`, scaled to full dataset)*",
        "",
    ]
    for mode in ["pairwise", "ranking", "pointwise"]:
        if mode not in ref_costs:
            continue
        mode_ref = ref_costs[mode]
        lines += [
            f"### {mode.capitalize()}",
            "",
            "| Model | Est. full cost |",
            "|---|---|",
        ]
        for model, cost in sorted(mode_ref["by_model"].items(), key=lambda x: -x[1]):
            lines.append(f"| `{model}` | ${cost:.4f} |")
        lines += [f"| **Total** | **${mode_ref['total']:.4f}** |", ""]
    lines.append("*est. = per-model reference cost; est. full = grand-total fallback (mode unknown)*")

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweeps-dir", default="output/ablation_sweeps")
    parser.add_argument("--ref", default=REFERENCE_COST_FILE)
    parser.add_argument("--retrieval-ref", default=RETRIEVAL_REFERENCE_COST_FILE)
    parser.add_argument("--date", default=None,
                        help="Filter to a specific date (YYYYMMDD or YYYY-MM-DD)")
    parser.add_argument("--out", default="output/ablation_cost_report.md",
                        help="Output path for the Markdown report")
    args = parser.parse_args()

    date_filter = args.date.replace("-", "") if args.date else None

    repo_root = Path(__file__).parent.parent
    sweeps_dir = (repo_root / args.sweeps_dir).resolve()
    ref_path = (repo_root / args.ref).resolve()
    retrieval_ref_path = (repo_root / args.retrieval_ref).resolve()
    out_path = (repo_root / args.out).resolve()

    if not sweeps_dir.is_dir():
        sys.exit(f"ERROR: sweeps dir not found: {sweeps_dir}")
    if not ref_path.is_file():
        sys.exit(f"ERROR: reference cost file not found: {ref_path}")
    if not retrieval_ref_path.is_file():
        sys.exit(f"ERROR: retrieval reference cost file not found: {retrieval_ref_path}")

    base_ref = load_reference_costs(ref_path)
    retrieval_ref = load_reference_costs(retrieval_ref_path)
    ok_entries, error_entries = collect_all_entries(sweeps_dir, base_ref, retrieval_ref, date_filter)

    if date_filter and not ok_entries and not error_entries:
        sys.exit(f"No data found for date {date_filter}")

    report = generate_report(ok_entries, error_entries, base_ref, date_filter)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"Report written to {out_path}")

    # quick summary to stdout
    total = sum(e.cost_usd for e in ok_entries if "FAIL" not in e.status.upper())
    print(f"Total estimated cost: ${total:.4f}")
    print(f"Read errors: {len(error_entries)}")


if __name__ == "__main__":
    main()
