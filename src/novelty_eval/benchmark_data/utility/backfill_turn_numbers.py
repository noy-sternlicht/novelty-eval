#!/usr/bin/env python3
"""
Backfill turn_number into the metadata of generated-negative entries in a pairwise or
pointwise dataset YAML.

For each NEGATIVE idea, looks up the brainstorming turn on which it was produced by
matching its abstract text against selected_research_plan.json files in the corresponding
_generation directory, then reading the sibling divergent_phase_log.json.

Supports both formats:
  - Pairwise: instance has "ideas" dict and nested "metadata" dict keyed by idea index.
  - Pointwise: instance has a single "idea" string and a flat "metadata" dict.

Usage:
    python backfill_turn_numbers.py \\
        --dataset output/benchmark_instances/pairwise_data/<ts>/benchmark_instances.yaml \\
        --generation-dir output/benchmark_instances/pairwise_data/<ts>_generation \\
        [--output path/to/output.yaml] \\
        [--dry-run]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

if "SECRETS" not in os.environ:
    _project_root = Path(__file__).resolve().parents[3]
    _secrets_file = _project_root / "secrets.toml"
    if _secrets_file.exists():
        os.environ["SECRETS"] = str(_secrets_file)

from logging_utils import setup_logger

LOGGER = setup_logger(output_dir="..", console_level="INFO")


def _plan_to_abstract(plan: dict) -> str:
    """Reconstruct abstract text from a plan dict using the same logic as _selected_plan_to_paper."""
    abstract = plan.get("abstract", "")
    if not abstract:
        parts = [
            f"**{field}**: {plan[field]}"
            for field in ("context", "purpose", "mechanism", "evaluation")
            if plan.get(field)
        ]
        abstract = "\n".join(parts)
    return abstract


def build_plan_index(generation_dir: Path) -> dict:
    """Return {abstract_text: {"plan_id": str, "log_path": Path}} for all plans in generation_dir."""
    index: dict = {}
    plan_files = list((generation_dir / "generated_negatives").rglob("selected_research_plan.json"))
    LOGGER.info(f"Found {len(plan_files)} selected_research_plan.json files in {generation_dir}")

    for plan_path in plan_files:
        with open(plan_path) as f:
            plan = json.load(f)

        abstract = _plan_to_abstract(plan)
        if not abstract:
            LOGGER.warning(f"Empty abstract for plan at {plan_path}, skipping")
            continue

        plan_id = plan.get("id", "")
        log_path = plan_path.parent / "divergent_phase_log.json"

        if abstract in index:
            LOGGER.warning(
                f"Duplicate abstract text found — keeping first occurrence. "
                f"Duplicate at: {plan_path}"
            )
            continue

        index[abstract] = {"plan_id": plan_id, "log_path": log_path}

    return index


def lookup_turn_number(plan_id: str, log_path: Path) -> int | None:
    """Return the turn_number for plan_id in the divergent phase log, or None if not found."""
    if not log_path.exists():
        LOGGER.warning(f"Log file not found: {log_path}")
        return None

    with open(log_path) as f:
        log = json.load(f)

    for entry in log:
        rp = (entry.get("action") or {}).get("research_plan") or {}
        if rp.get("id") == plan_id:
            return entry["turn_number"]

    LOGGER.warning(f"Plan id '{plan_id}' not found in {log_path}")
    return None


def _patch_instance(inst_id, instance, plan_index, dry_run) -> tuple[int, int, int]:
    """Patch a single instance in-place. Returns (patched, already_present, not_found)."""
    patched = already_present = not_found = 0

    # Pairwise: ideas is a dict of {idx: abstract}
    if instance.get("ideas") is not None:
        ideas = instance["ideas"]
        metadata = instance.get("metadata", {})
        for idx, meta in metadata.items():
            if meta.get("type") != "NEGATIVE":
                continue
            if "turn_number" in meta and "divergence_log_path" in meta:
                already_present += 1
                continue
            abstract = ideas.get(idx, "")
            entry = plan_index.get(abstract)
            if entry is None:
                LOGGER.warning(f"Instance {inst_id}, idea {idx}: no matching plan found in index")
                not_found += 1
                continue
            turn_number = lookup_turn_number(entry["plan_id"], entry["log_path"])
            if turn_number is None:
                not_found += 1
                continue
            if dry_run:
                LOGGER.info(
                    f"[dry-run] Instance {inst_id}, idea {idx} "
                    f"(title: {meta.get('title', 'N/A')}): turn_number = {turn_number}, "
                    f"log = {entry['log_path']}"
                )
            else:
                meta["turn_number"] = turn_number
                meta["divergence_log_path"] = str(entry["log_path"])
            patched += 1

    # Pointwise: single idea string with top-level label
    elif instance.get("idea") is not None:
        if instance.get("label") != "NEGATIVE":
            return patched, already_present, not_found
        meta = instance.get("metadata", {})
        if "turn_number" in meta and "divergence_log_path" in meta:
            return 0, 1, 0
        abstract = instance["idea"]
        entry = plan_index.get(abstract)
        if entry is None:
            LOGGER.warning(f"Instance {inst_id}: no matching plan found in index")
            return 0, 0, 1
        turn_number = lookup_turn_number(entry["plan_id"], entry["log_path"])
        if turn_number is None:
            return 0, 0, 1
        if dry_run:
            LOGGER.info(
                f"[dry-run] Instance {inst_id} "
                f"(title: {meta.get('title', 'N/A')}): turn_number = {turn_number}, "
                f"log = {entry['log_path']}"
            )
        else:
            meta["turn_number"] = turn_number
            meta["divergence_log_path"] = str(entry["log_path"])
        patched = 1

    return patched, already_present, not_found


_INHERIT_FIELDS = ("turn_number", "divergence_log_path")


def _build_source_text_index(source: dict) -> dict:
    """Build {idea_text: {field: value}} from all NEGATIVE ideas in a source dataset.

    Handles both pairwise (ideas dict) and pointwise (single idea) source formats.
    Emits a warning when two entries share identical idea text.
    """
    index: dict = {}
    for inst in source.values():
        if inst.get("ideas") is not None:  # pairwise source
            for idx, meta in inst.get("metadata", {}).items():
                if meta.get("type") != "NEGATIVE":
                    continue
                idea_text = inst["ideas"].get(idx, "")
                values = {f: meta[f] for f in _INHERIT_FIELDS if f in meta}
        elif inst.get("idea") is not None:  # pointwise source
            if inst.get("label") != "NEGATIVE":
                continue
            idea_text = inst["idea"]
            values = {f: inst["metadata"][f] for f in _INHERIT_FIELDS if f in inst.get("metadata", {})}
        else:
            continue

        if not idea_text or not values:
            continue
        if idea_text in index:
            LOGGER.warning("Duplicate idea text in source dataset — keeping first occurrence")
            continue
        index[idea_text] = values
    return index


def _patch_from_source(dataset: dict, source: dict, dry_run: bool) -> tuple[int, int, int]:
    """Copy turn_number and divergence_log_path from source into dataset.

    Strategy depends on the target format:
    - Pairwise target: matches by inst_id + idea index (works when source and target
      share the same instance structure, e.g. a manipulated derivative of the source).
    - Pointwise target: matches by idea text against all NEGATIVE ideas in the source
      (necessary because pointwise instances are re-indexed after shuffling).

    Source must already have the fields populated (e.g. previously backfilled via --generation-dir).
    """
    patched = already_present = not_found = 0

    sample_inst = next(iter(dataset.values()))
    target_is_pointwise = sample_inst.get("idea") is not None

    if target_is_pointwise:
        text_index = _build_source_text_index(source)
        LOGGER.info(f"Source text index built: {len(text_index)} NEGATIVE entries")
        for inst_id, instance in dataset.items():
            if instance.get("label") != "NEGATIVE":
                continue
            meta = instance.get("metadata", {})
            if all(f in meta for f in _INHERIT_FIELDS):
                already_present += 1
                continue
            values = text_index.get(instance["idea"])
            if not values:
                LOGGER.warning(f"Instance {inst_id}: no match in source text index")
                not_found += 1
                continue
            if dry_run:
                LOGGER.info(
                    f"[dry-run] Instance {inst_id} "
                    f"(title: {meta.get('title', 'N/A')}): {values}"
                )
            else:
                meta.update(values)
            patched += 1
        return patched, already_present, not_found

    # Pairwise target: match by inst_id + idea index
    for inst_id, instance in dataset.items():
        src_instance = source.get(inst_id)
        if src_instance is None:
            LOGGER.warning(f"Instance {inst_id} not found in source dataset, skipping")
            not_found += 1
            continue

        if instance.get("ideas") is not None:
            metadata = instance.get("metadata", {})
            src_metadata = src_instance.get("metadata", {})
            for idx, meta in metadata.items():
                if meta.get("type") != "NEGATIVE":
                    continue
                if all(f in meta for f in _INHERIT_FIELDS):
                    already_present += 1
                    continue
                src_meta = src_metadata.get(idx, {})
                values = {f: src_meta[f] for f in _INHERIT_FIELDS if f in src_meta}
                if not values:
                    LOGGER.warning(
                        f"Instance {inst_id}, idea {idx}: source metadata missing {_INHERIT_FIELDS}"
                    )
                    not_found += 1
                    continue
                if dry_run:
                    LOGGER.info(
                        f"[dry-run] Instance {inst_id}, idea {idx} "
                        f"(title: {meta.get('title', 'N/A')}): {values}"
                    )
                else:
                    meta.update(values)
                patched += 1

        # Pointwise source within a pairwise-target loop (shouldn't normally occur,
        # but handle gracefully via text index as fallback)
        elif instance.get("idea") is not None:
            if instance.get("label") != "NEGATIVE":
                continue
            meta = instance.get("metadata", {})
            if all(f in meta for f in _INHERIT_FIELDS):
                already_present += 1
                continue
            src_meta = src_instance.get("metadata", {})
            values = {f: src_meta[f] for f in _INHERIT_FIELDS if f in src_meta}
            if not values:
                LOGGER.warning(f"Instance {inst_id}: source metadata missing {_INHERIT_FIELDS}")
                not_found += 1
                continue
            if dry_run:
                LOGGER.info(
                    f"[dry-run] Instance {inst_id} "
                    f"(title: {meta.get('title', 'N/A')}): {values}"
                )
            else:
                meta.update(values)
            patched += 1

    return patched, already_present, not_found


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill turn_number and divergence_log_path into NEGATIVE metadata entries "
                    "of a pairwise or pointwise YAML."
    )
    parser.add_argument("--dataset", required=True, help="Path to benchmark_instances.yaml or iclr_pointwise_instances.yaml")

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--generation-dir",
        help="Path to the *_generation directory that produced the negatives (matches by abstract text)",
    )
    source_group.add_argument(
        "--source-dataset",
        help="Path to a parent dataset YAML that already has turn_number/divergence_log_path "
             "(matches by instance/idea index, for post-processed derivatives)",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output YAML path (default: overwrite in-place)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing the file",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        LOGGER.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    with open(dataset_path) as f:
        dataset: dict = yaml.safe_load(f)

    patched = already_present = not_found = 0

    if args.generation_dir:
        generation_dir = Path(args.generation_dir).resolve()
        if not generation_dir.exists():
            LOGGER.error(f"Generation dir not found: {generation_dir}")
            sys.exit(1)
        plan_index = build_plan_index(generation_dir)
        LOGGER.info(f"Plan index built: {len(plan_index)} unique abstracts")
        for inst_id, instance in dataset.items():
            p, a, n = _patch_instance(inst_id, instance, plan_index, args.dry_run)
            patched += p
            already_present += a
            not_found += n
    else:
        source_path = Path(args.source_dataset).resolve()
        if not source_path.exists():
            LOGGER.error(f"Source dataset not found: {source_path}")
            sys.exit(1)
        with open(source_path) as f:
            source: dict = yaml.safe_load(f)
        LOGGER.info(f"Loaded source dataset: {source_path} ({len(source)} instances)")
        patched, already_present, not_found = _patch_from_source(dataset, source, args.dry_run)

    LOGGER.info(
        f"Summary — patched: {patched}, already present: {already_present}, not found: {not_found}"
    )

    if args.dry_run:
        LOGGER.info("Dry-run mode: no changes written.")
        return

    output_path = Path(args.output).resolve() if args.output else dataset_path
    with open(output_path, "w") as f:
        yaml.dump(dataset, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    LOGGER.info(f"Patched YAML written to: {output_path}")


if __name__ == "__main__":
    main()
