#!/usr/bin/env python3
"""
Playground for experimenting with the extract_novelty_signals.jinja2 prompt.

Usage examples:
  # Run on a random review from the ICLR 2026 dataset
  python extract_novelty_signals_playground.py

  # Run on N random papers and show results for each
  python extract_novelty_signals_playground.py --n 3

  # Paste your review sections directly into CUSTOM_STRENGTHS_REVIEW below, then run:
  python extract_novelty_signals_playground.py

  # Try a different model
  python extract_novelty_signals_playground.py --model gpt-4o

  # Write results to a JSON file
  python extract_novelty_signals_playground.py --n 5 --output results.json
"""

import argparse
import json
import os
import random
import sys
import textwrap
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

if "SECRETS" not in os.environ:
    project_root = Path(__file__).resolve().parents[3]
    secrets_file = project_root / "secrets.toml"
    if secrets_file.exists():
        os.environ["SECRETS"] = str(secrets_file)

from utils import extract_json_choice, prompt_openai_client  # noqa: E402

# ── paths ─────────────────────────────────────────────────────────────────────
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
DEFAULT_TEMPLATE_NAME = "extract_novelty_signals.jinja2"
ICLR_DATA_PATH = Path(__file__).parent / "iclr_data" / "iclr_2026_data.json"

# ── ANSI colours ──────────────────────────────────────────────────────────────
RED   = "\033[31m"
GREEN = "\033[32m"
CYAN  = "\033[36m"
BOLD  = "\033[1m"
RESET = "\033[0m"

# ── Custom reviews ────────────────────────────────────────────────────────────
# Paste your own strengths/weaknesses here to test them without any CLI flags.
# Leave as None (or empty string) to use random papers from the ICLR dataset.
CUSTOM_STRENGTHS_REVIEW = """
I feel the paper is written well and also motivated well. The internal mechanisms of reasoning traces is worth looking into, although I have doubts regarding the experimental sections. Please see weaknesses.
"""

CUSTOM_WEAKNESSES_REVIEW = """
1. Just AIME 2024/25 is too small for conclusion to be drawn
2. The overall method seems like a variant of self-consistency combined with MCTS and majority voting. SC is already known to be better than vanilla CoT. I think just comparing with A_last does not give enough insights, one needs to look at other strong baselines such as SC-CoT
3. a relatively less concerning weakness is heuristic based segmentation
"""

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_template(name: str = DEFAULT_TEMPLATE_NAME):
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    return env.get_template(name)


def load_iclr_data() -> list:
    with open(ICLR_DATA_PATH) as f:
        return json.load(f)


def pick_reviews(data: list, n: int) -> list:
    """Return a list of n random paper/review pairs."""
    selected = []
    # Flatten the papers/reviews to easily pick n reviews
    all_reviews = []
    for paper in data:
        title = paper.get("title", "Unknown")
        for review in paper.get("reviews", []):
            all_reviews.append((title, review))
    
    return random.sample(all_reviews, min(n, len(all_reviews)))


def run_prompt(template, strengths: str, weaknesses: str, model: str) -> str:
    prompt = template.render(
        strengths_review=strengths,
        weaknesses_review=weaknesses,
    )
    return prompt_openai_client(prompt, engine=model), prompt


def _wrap(text: str, indent: str = "  ", subsequent_indent: str = None) -> str:
    if not text:
        return f"{indent}(empty)"
    if subsequent_indent is None:
        subsequent_indent = indent
    return textwrap.fill(text.strip(), width=80, initial_indent=indent, subsequent_indent=subsequent_indent)


def display_result(title: str, strengths: str, weaknesses: str, parsed: dict, raw_response: str):
    sep  = "=" * 80
    thin = "-" * 80

    print(f"\n{BOLD}{CYAN}{sep}{RESET}")
    print(f"{BOLD}📄 Paper:{RESET} {title}")
    print(f"{CYAN}{sep}{RESET}")

    print(f"\n{BOLD}💪 Strengths Review (Input):{RESET}")
    print(thin)
    print(_wrap(strengths))

    print(f"\n{BOLD}⚠️  Weaknesses Review (Input):{RESET}")
    print(thin)
    print(_wrap(weaknesses))

    print(f"\n{BOLD}🎯 Extracted Novelty Signals (Parsed Output):{RESET}")
    print(thin)
    if parsed:
        # Support both template formats:
        #   v0:      strengths_novelty_sentences / weaknesses_novelty_sentences
        #   current: positive_novelty_snippets   / negative_novelty_snippets
        pos_key = "positive_novelty_snippets" if "positive_novelty_snippets" in parsed else "strengths_novelty_sentences"
        neg_key = "negative_novelty_snippets" if "negative_novelty_snippets" in parsed else "weaknesses_novelty_sentences"

        print(f"{BOLD}  • Positive Novelty Snippets:{RESET}")
        for s in parsed.get(pos_key, []):
            wrapped_s = _wrap(s, indent="    - ", subsequent_indent="      ")
            print(f"{GREEN}{wrapped_s}{RESET}")
        if not parsed.get(pos_key):
            print(f"    {RED}(None){RESET}")

        print(f"\n{BOLD}  • Negative Novelty Snippets:{RESET}")
        for s in parsed.get(neg_key, []):
            wrapped_s = _wrap(s, indent="    - ", subsequent_indent="      ")
            print(f"{RED}{wrapped_s}{RESET}")
        if not parsed.get(neg_key):
            print(f"    {GREEN}(None){RESET}")

        print(f"\n{BOLD}  • Similar Papers Mentioned:{RESET}")
        for p in parsed.get("similar_papers_mentioned", []):
            wrapped_p = _wrap(p, indent="    - ", subsequent_indent="      ")
            print(f"{CYAN}{wrapped_p}{RESET}")
        if not parsed.get("similar_papers_mentioned"):
            print(f"    (None)")
    else:
        print(f"{RED}Failed to parse JSON output. Raw response below:{RESET}")
        print(raw_response)

    print(f"\n{CYAN}{sep}{RESET}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Playground for extract_novelty_signals.jinja2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--n",           type=int,  default=1,        help="Number of random reviews to process (default: 1).")
    parser.add_argument("--model",       type=str,  default="claude-opus-4-6", help="LLM model to use.")
    parser.add_argument("--template",    type=str,  default=DEFAULT_TEMPLATE_NAME, help=f"Template filename to use from templates/ dir (default: {DEFAULT_TEMPLATE_NAME}). Use 'extract_novelty_signals_v0.jinja2' for the original prompt.")
    parser.add_argument("--strengths",   type=str,  default=None,     help="Override strengths for custom run.")
    parser.add_argument("--weaknesses",  type=str,  default=None,     help="Override weaknesses for custom run.")
    parser.add_argument("--show-prompt", action="store_true",         help="Print rendered prompt before model call.")
    parser.add_argument("--output",      type=str,  default=None,     help="Optional path to save results as JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    template = load_template(args.template)
    results  = []

    # ── Override or Custom globals ────────────────────────────────────────────
    if args.strengths or args.weaknesses or CUSTOM_STRENGTHS_REVIEW.strip() or CUSTOM_WEAKNESSES_REVIEW.strip():
        strengths = args.strengths or CUSTOM_STRENGTHS_REVIEW.strip()
        weaknesses = args.weaknesses or CUSTOM_WEAKNESSES_REVIEW.strip()
        
        print(f"{BOLD}Running on CUSTOM/OVERRIDE input with model '{args.model}'...{RESET}")
        raw_response, prompt = run_prompt(template, strengths, weaknesses, args.model)
        print(raw_response)

        if args.show_prompt:
            print(f"\n{BOLD}--- Rendered Prompt ---{RESET}\n{prompt}\n{BOLD}-----------------------{RESET}\n")
            
        parsed = extract_json_choice(raw_response)
        print(parsed)
        display_result("(custom review)", strengths, weaknesses, parsed, raw_response)
        results.append({
            "title": "(custom)",
            "strengths": strengths,
            "weaknesses": weaknesses,
            "raw_response": raw_response,
            "parsed": parsed
        })

    # ── from ICLR dataset ─────────────────────────────────────────────────────
    else:
        data   = load_iclr_data()
        pairs = pick_reviews(data, args.n)
        print(f"{BOLD}Running on {len(pairs)} random review(s) with model '{args.model}'...{RESET}")

        for title, review in pairs:
            strengths = review.get("strengths", {}).get("value", "")
            weaknesses = review.get("weaknesses", {}).get("value", "")
            
            if not strengths and not weaknesses:
                print(f"{RED}Skipping review for '{title}' — no content found.{RESET}")
                continue

            print(f"\n⏳ Processing review for: {BOLD}{title[:80]}{'...' if len(title) > 80 else ''}{RESET}")
            raw_response, prompt = run_prompt(template, strengths, weaknesses, args.model)
            parsed = extract_json_choice(raw_response)
            display_result(title, strengths, weaknesses, parsed, raw_response)
            results.append({
                "title": title,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "raw_response": raw_response,
                "parsed": parsed
            })

    # ── optional JSON output ──────────────────────────────────────────────────
    if args.output and results:
        out_path = Path(args.output)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"{GREEN}Results saved to {out_path.resolve()}{RESET}")


if __name__ == "__main__":
    main()
