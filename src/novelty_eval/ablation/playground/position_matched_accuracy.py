#!/usr/bin/env python3
"""Pairwise accuracy on all instances vs. on the position-matched subset.

The hvh and vanilla-ai tracks hold the same 161 positives but assign the gold
winner to slot 0 or slot 1 independently, so a unidirectional judge (mec_k=1,
bidirectional=false) sees a different position mix in each track.  This script
quantifies how much that matters: it recomputes each run's accuracy restricted
to the instances whose gold-winner *position* agrees across the two tracks, and
reports the delta against the full-data accuracy.

Instances are joined across tracks on the POSITIVE PAPER TITLE, not on
problem_id -- the two tracks are ordered differently and share only 19/161
problem_ids, so ids are not interchangeable.

Usage:
    python position_matched_accuracy.py [--csv]
"""
import argparse
import json
import sys
from pathlib import Path

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:  # pragma: no cover - PyYAML without libyaml
    from yaml import SafeLoader
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[4]

# --- what to report ------------------------------------------------------- #
# Instance files, per track (must be the ones the sweeps below actually ran on).
INSTANCES = {
    "hvh": "data/human-only/pairwise.yaml",
    "vanilla-ai": "data/human-plus-generated/pairwise.yaml",
    "hvh-plan": "data/human-only/pairwise_plan.yaml",
    "vanilla-ai-plan": "data/human-plus-generated/pairwise_plan.yaml",
}

# One sweep dir per track.  These four are contemporaneous (2026-07-24) and each
# covers all four ai_researcher_* ablations x both judges.
SWEEPS = {
    "hvh": "output/ablation_sweeps/20260724_130032",
    "vanilla-ai": "output/ablation_sweeps/20260724_130142",
    "hvh-plan": "output/ablation_sweeps/20260724_125924",
    "vanilla-ai-plan": "output/ablation_sweeps/20260724_120203",
}

# The two tracks whose position agreement defines the matched subset.  The -plan
# variants inherit their bases' positions, so one matched set covers all four.
MATCH_PAIR = ("hvh", "vanilla-ai")

TRACK_ORDER = ["hvh", "vanilla-ai", "hvh-plan", "vanilla-ai-plan"]
ABLATIONS = {
    "ai_researcher_base": "base",
    "ai_researcher_conference_mention": "no conference mention",
    "ai_researcher_no_review": "no review mention",
    "ai_researcher_comparative": "comparative",
}
JUDGES = ["claude-opus-4-6", "gpt-5.4"]


# --- instance helpers ----------------------------------------------------- #
def load_instances(track: str) -> dict:
    with open(_PROJECT_ROOT / INSTANCES[track]) as fh:
        return yaml.load(fh, Loader=SafeLoader)


def gold_position(entry: dict) -> str:
    """The slot ('0'/'1') holding the expected winner."""
    winners = entry["expected_winners"]
    if len(winners) != 1:
        raise ValueError(f"expected exactly one winner, got {winners}")
    return str(winners[0])


def gold_title(entry: dict) -> str:
    """Title of the positive paper -- the only cross-track join key."""
    pos = gold_position(entry)
    metadata = entry["metadata"]
    # metadata keys are str in some instance files and int in others
    return metadata.get(pos, metadata.get(int(pos)))["title"]


def positions_by_title(track: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in load_instances(track).values():
        title = gold_title(entry)
        if title in out:
            raise ValueError(f"duplicate positive title in {track}: {title!r}")
        out[title] = gold_position(entry)
    return out


# --- run helpers ---------------------------------------------------------- #
def load_run(track: str, ablation: str, judge: str) -> dict[str, float]:
    """title -> 1.0 correct / 0.0 wrong / 0.5 tie, for one sweep cell."""
    experiment = f"pairwise-{track}_{ablation}"
    pattern = f"{experiment}/{experiment}-{judge}/accuracy_test_artifacts/*/run_pairwise_*/scores.json"
    hits = sorted((_PROJECT_ROOT / SWEEPS[track]).glob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(
            f"expected exactly 1 scores.json for {experiment}-{judge}, found {len(hits)}"
        )

    instances = load_instances(track)
    outcomes: dict[str, float] = {}
    for problem_id, problem in json.loads(hits[0].read_text()).items():
        comparisons = problem.get("comparisons") or []
        if not comparisons:
            continue
        if len(comparisons) != 1:
            raise ValueError(f"{experiment}: expected 1 comparison, got {len(comparisons)}")
        comp = comparisons[0]

        entry = instances.get(int(problem_id), instances.get(str(problem_id)))
        gold = gold_position(entry)
        winner = comp.get("winner")
        if winner == 0:
            picked = str(comp["idea_0"])
        elif winner == 1:
            picked = str(comp["idea_1"])
        else:  # tie or unparsable verdict
            picked = None

        # Ties count as 0.5, matching calculate_pairwise_accuracy's
        # `accuracy_with_ties` convention.
        outcomes[gold_title(entry)] = 0.5 if picked is None else float(picked == gold)
    return outcomes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="store_true", help="emit CSV instead of a markdown table")
    args = parser.parse_args()

    left, right = MATCH_PAIR
    pos_left, pos_right = positions_by_title(left), positions_by_title(right)
    titles = sorted(set(pos_left) & set(pos_right))
    matched = {t for t in titles if pos_left[t] == pos_right[t]}
    if not titles:
        print(f"no shared positives between {left} and {right}", file=sys.stderr)
        return 1

    print(
        f"# joined on positive title: {len(titles)} shared, "
        f"{len(matched)} position-matched, {len(titles) - len(matched)} unmatched "
        f"(match defined by {left} vs {right})"
    )

    header = ["judge", "ablation", "track", f"all (n={len(titles)})",
              f"matched (n={len(matched)})", "delta"]
    rows = []
    deltas = []  # (delta, "judge/ablation/track") for the summary below
    for judge in JUDGES:
        for ablation, label in ABLATIONS.items():
            for track in TRACK_ORDER:
                outcomes = load_run(track, ablation, judge)
                missing = [t for t in titles if t not in outcomes]
                if missing:
                    raise ValueError(
                        f"pairwise-{track}_{ablation}-{judge}: {len(missing)} shared "
                        f"positives absent from the run (e.g. {missing[0]!r})"
                    )
                all_acc = sum(outcomes[t] for t in titles) / len(titles)
                mat_acc = sum(outcomes[t] for t in matched) / len(matched)
                delta = mat_acc - all_acc
                deltas.append((delta, f"{judge}/{label}/{track}"))
                rows.append([judge, label, track,
                             f"{all_acc:.3f}", f"{mat_acc:.3f}", f"{delta:+.3f}"])

    if args.csv:
        print(",".join(header))
        for row in rows:
            print(",".join(row))
    else:
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        for row in rows:
            print("| " + " | ".join(row) + " |")

    # Summary, in accuracy points (delta * 100).  Mean |delta| answers "how much
    # does the subset move a number on average", max |delta| bounds the worst case.
    mean_abs = 100 * sum(abs(d) for d, _ in deltas) / len(deltas)
    mean_signed = 100 * sum(d for d, _ in deltas) / len(deltas)
    worst, worst_cell = max(deltas, key=lambda item: abs(item[0]))
    print()
    print(f"# {len(deltas)} cells | mean |delta| = {mean_abs:.2f} accuracy points "
          f"(mean signed {mean_signed:+.2f}) | max |delta| = {abs(100 * worst):.2f} "
          f"points ({100 * worst:+.2f}, {worst_cell})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
