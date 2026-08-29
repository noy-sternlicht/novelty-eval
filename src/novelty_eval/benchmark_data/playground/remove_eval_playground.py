#!/usr/bin/env python3
"""
Playground for experimenting with the remove_eval_data.jinja2 prompt.

Usage examples:
  # Run on a random abstract from the ICLR 2026 dataset
  python remove_eval_playground.py

  # Run on N random papers and show a diff for each
  python remove_eval_playground.py --n 3

  # Paste your abstract directly into CUSTOM_ABSTRACT below, then run:
  python remove_eval_playground.py

  # Try a different model
  python remove_eval_playground.py --model gpt-4o

  # Write results to a JSON file
  python remove_eval_playground.py --n 5 --output results.json
"""

import argparse
import difflib
import json
import os
import random
import sys
import textwrap
from pathlib import Path

from jinja2 import Template

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

if "SECRETS" not in os.environ:
    project_root = Path(__file__).resolve().parents[3]
    secrets_file = project_root / "secrets.toml"
    if secrets_file.exists():
        os.environ["SECRETS"] = str(secrets_file)

from utils import prompt_openai_client  # noqa: E402  (after path setup)

# ── paths ─────────────────────────────────────────────────────────────────────
TEMPLATE_PATH = Path(__file__).parent / "templates" / "remove_eval_data.jinja2"
ICLR_DATA_PATH = Path(__file__).parent / "iclr_data" / "iclr_2026_data.json"

# ── ANSI colours ──────────────────────────────────────────────────────────────
RED   = "\033[31m"
GREEN = "\033[32m"
CYAN  = "\033[36m"
BOLD  = "\033[1m"
RESET = "\033[0m"

# ── Custom abstract ───────────────────────────────────────────────────────────
# Paste your own abstract here to test it without any CLI flags.
# Leave as None (or empty string) to use random papers from the ICLR dataset.
# CUSTOM_ABSTRACT = """
# A hallmark of human innovation is recombination -- the creation of novel ideas by integrating elements from existing concepts and mechanisms. In this work, we introduce CHIMERA, a large-scale Knowledge Base (KB) of over 28K recombination examples automatically mined from the scientific literature. CHIMERA enables large-scale empirical analysis of how scientists recombine concepts and draw inspiration from different areas, and enables training models that propose novel, cross-disciplinary research directions. To construct this KB, we define a new information extraction task: identifying recombination instances in scientific abstracts. We curate a high-quality, expert-annotated dataset and use it to fine-tune a large language model, which we apply to a broad corpus of AI papers. We showcase the utility of CHIMERA through two applications. First, we analyze patterns of recombination across AI subfields. Second, we train a scientific hypothesis generation model using the KB, showing that it can propose novel research directions that researchers rate as inspiring. We release our data and code at this https URL.
# """
CUSTOM_ABSTRACT = """
# We introduce Debate Speech Evaluation as a novel and challenging benchmark for assessing LLM judges. Evaluating debate speeches requires a deep understanding of the speech at multiple levels, including argument strength and relevance, the coherence and organization of the speech, the appropriateness of its style and tone, and so on. This task involves a unique set of cognitive abilities that previously received limited attention in systematic LLM benchmarking. To explore such skills, we leverage a dataset of over 600 meticulously annotated debate speeches and present the first in-depth analysis of how state-of-the-art LLMs compare to human judges on this task. Our findings reveal a nuanced picture: while larger models can approximate individual human judgments in some respects, they differ substantially in their overall judgment behavior. We also investigate the ability of frontier LLMs to generate persuasive, opinionated speeches, showing that models may perform at a human level on this task.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_template() -> Template:
    with open(TEMPLATE_PATH) as f:
        return Template(f.read())


def load_iclr_data() -> list:
    with open(ICLR_DATA_PATH) as f:
        return json.load(f)


def pick_papers(data: list, n: int) -> list:
    """Return a list of n random paper dicts."""
    return random.sample(data, min(n, len(data)))


def run_prompt(template: Template, abstract: str, model: str) -> str:
    prompt = template.render(abstract=abstract)
    return prompt_openai_client(prompt, engine=model)


def _wrap(text: str, indent: str = "  ") -> str:
    return textwrap.fill(text.strip(), width=80, initial_indent=indent, subsequent_indent=indent)


def word_diff(original: str, modified: str) -> str:
    """Word-level diff: removed words in red, added in green."""
    orig_words = original.split()
    mod_words  = modified.split()
    matcher = difflib.SequenceMatcher(None, orig_words, mod_words, autojunk=False)
    parts = []
    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == "equal":
            parts.append(" ".join(orig_words[a0:a1]))
        elif opcode == "replace":
            parts.append(f"{RED}{' '.join(orig_words[a0:a1])}{RESET}")
            parts.append(f"{GREEN}{' '.join(mod_words[b0:b1])}{RESET}")
        elif opcode == "delete":
            parts.append(f"{RED}{' '.join(orig_words[a0:a1])}{RESET}")
        elif opcode == "insert":
            parts.append(f"{GREEN}{' '.join(mod_words[b0:b1])}{RESET}")
    return textwrap.fill(" ".join(parts), width=80, initial_indent="  ", subsequent_indent="  ")


def display_result(title: str, original: str, modified: str, show_diff: bool):
    sep  = "=" * 80
    thin = "-" * 80

    print(f"\n{BOLD}{CYAN}{sep}{RESET}")
    print(f"{BOLD}📄 Paper:{RESET} {title}")
    print(f"{CYAN}{sep}{RESET}")

    print(f"\n{BOLD}📝 Original Abstract:{RESET}")
    print(thin)
    print(_wrap(original))

    print(f"\n{BOLD}✂️  Modified Abstract (evaluation removed):{RESET}")
    print(thin)
    print(_wrap(modified))

    if show_diff:
        print(f"\n{BOLD}🔍 Word-Level Diff{RESET}  {RED}red = removed{RESET}  ·  {GREEN}green = added{RESET}")
        print(thin)
        print(word_diff(original, modified))

    print(f"\n{CYAN}{sep}{RESET}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Playground for remove_eval_data.jinja2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--n",       type=int,  default=1,        help="Number of random papers to process (default: 1).")
    parser.add_argument("--model",   type=str,  default="claude-opus-4-6", help="LLM model to use.")
    parser.add_argument("--no-diff", action="store_true",         help="Skip the word-level diff display.")
    parser.add_argument("--output",  type=str,  default=None,     help="Optional path to save results as JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    template = load_template()
    results  = []

    # ── CUSTOM_ABSTRACT global takes priority ─────────────────────────────────
    custom = CUSTOM_ABSTRACT.strip()
    if custom:
        print(f"{BOLD}Running on CUSTOM_ABSTRACT with model '{args.model}'...{RESET}")
        modified = run_prompt(template, custom, args.model)
        display_result("(custom abstract)", custom, modified, not args.no_diff)
        results.append({"title": "(custom)", "original": custom, "modified": modified})

    # ── from ICLR dataset ─────────────────────────────────────────────────────
    else:
        data   = load_iclr_data()
        papers = pick_papers(data, args.n)
        print(f"{BOLD}Running on {len(papers)} paper(s) with model '{args.model}'...{RESET}")

        for paper in papers:
            title    = paper.get("title", "Unknown")
            abstract = paper.get("abstract", "")
            if not abstract:
                print(f"{RED}Skipping '{title}' — no abstract found.{RESET}")
                continue

            print(f"\n⏳ Processing: {BOLD}{title[:80]}{'...' if len(title) > 80 else ''}{RESET}")
            modified = run_prompt(template, abstract, args.model)
            display_result(title, abstract, modified, not args.no_diff)
            results.append({"title": title, "original": abstract, "modified": modified})

    # ── optional JSON output ──────────────────────────────────────────────────
    if args.output and results:
        out_path = Path(args.output)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"{GREEN}Results saved to {out_path.resolve()}{RESET}")


if __name__ == "__main__":
    main()




