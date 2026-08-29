"""
Count occurrences of the _select_candidates_per_query ceil-overshoot bug in a run.log.

Bug signature:
  - Summary line shows N/N selected (full top_k allocation)
  - At least one per-query line shows viable_count > 0 but selected_count == 0

This is an upper-bound count: a few edge cases (top_k < n_queries) could produce
the same pattern legitimately, but those are rare in practice.

Usage:
    python scripts/count_retrieval_bug.py <path/to/run.log>
    python scripts/count_retrieval_bug.py <path/to/run.log> --verbose
"""

import re
import sys
import argparse
from dataclasses import dataclass, field
from pathlib import Path


SUMMARY_RE = re.compile(
    r"retrieve_candidates\.py:\d+\] Candidates: .* → (\d+)/(\d+) selected"
)
QUERY_RE = re.compile(
    r"^\s+'(.+?)': (.+) → (\d+) selected$"
)


@dataclass
class IdeaResult:
    line_no: int
    selected: int
    top_k: int
    n_queries: int = 0
    bug_queries: list = field(default_factory=list)  # (query_text, viable_count)

    @property
    def full_allocation(self) -> bool:
        return self.selected == self.top_k

    @property
    def is_bug_affected(self) -> bool:
        return self.full_allocation and bool(self.bug_queries)


def parse_log(path: Path) -> list[IdeaResult]:
    results: list[IdeaResult] = []
    current: IdeaResult | None = None

    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            # Check for summary line
            m = SUMMARY_RE.search(line)
            if m:
                current = IdeaResult(
                    line_no=lineno,
                    selected=int(m.group(1)),
                    top_k=int(m.group(2)),
                )
                results.append(current)
                continue

            # Check for per-query line (only meaningful when inside a block)
            if current is None:
                continue
            m = QUERY_RE.match(line)
            if not m:
                # Non-query line ends the block
                current = None
                continue

            query_text = m.group(1).strip("'").strip()
            # All numbers before the final "selected" count
            numbers_str = m.group(2)
            selected_count = int(m.group(3))
            # viable = last number in the chain (second-to-last overall)
            chain = [int(x) for x in numbers_str.split("→") if x.strip().isdigit()]
            viable_count = chain[-1] if chain else 0

            current.n_queries += 1
            if viable_count > 0 and selected_count == 0 and current.full_allocation:
                current.bug_queries.append((query_text, viable_count))

    return results


def main():
    parser = argparse.ArgumentParser(description="Count retrieval bug occurrences in run.log")
    parser.add_argument("log_path", type=Path, help="Path to run.log")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print each affected idea")
    args = parser.parse_args()

    if not args.log_path.exists():
        print(f"Error: {args.log_path} not found", file=sys.stderr)
        sys.exit(1)

    results = parse_log(args.log_path)

    total = len(results)
    insufficient = sum(1 for r in results if not r.full_allocation)
    full_alloc = sum(1 for r in results if r.full_allocation)
    bug_affected = sum(1 for r in results if r.is_bug_affected)
    total_bug_queries = sum(len(r.bug_queries) for r in results)

    print(f"Log: {args.log_path}")
    print(f"{'─' * 60}")
    print(f"Total retrieval blocks:     {total:>4}")
    print(f"  Insufficient candidates:  {insufficient:>4}  (skipped — can't tell)")
    print(f"  Full allocation (S==top_k): {full_alloc:>4}")
    print(f"{'─' * 60}")
    print(f"Bug-affected ideas:         {bug_affected:>4} / {full_alloc}  ({bug_affected/full_alloc*100:.1f}% of full-alloc)")
    print(f"Total starved queries:      {total_bug_queries:>4}  (queries with viable>0 but 0 selected)")
    if bug_affected:
        avg = total_bug_queries / bug_affected
        print(f"Avg starved queries/idea:   {avg:>7.2f}")
    print(f"{'─' * 60}")
    print(f"Note: these are upper-bound counts — a tiny fraction may be")
    print(f"legitimate (e.g. top_k < n_queries), but the vast majority")
    print(f"reflect the ceil-overshoot bug.")

    if args.verbose and bug_affected:
        print(f"\n{'─' * 60}")
        print("Affected ideas:")
        for r in results:
            if not r.is_bug_affected:
                continue
            print(f"\n  Line {r.line_no}: {r.selected}/{r.top_k} selected, {r.n_queries} queries")
            for qtext, viable in r.bug_queries:
                print(f"    STARVED  viable={viable:>3}  '{qtext}'")


if __name__ == "__main__":
    main()
