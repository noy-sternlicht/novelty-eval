#!/usr/bin/env python3
import argparse
import asyncio
import os
import sys
import yaml
import json
from typing import Dict, Any, List

from jinja2 import Environment, FileSystemLoader

# Add src to path to allow imports
# This adds the parent directory of novelty_eval (which is src) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import LOGGER, prompt_openai_client, extract_json_choice
from paper_finder_api import search_papers
from semantic_scholar import SEMANTIC_API

# ==============================================================================
# PROMPT PLAYGROUND AREA
# Edit the .j2 template files in the templates/ directory to experiment with
# query generation strategies.
#   templates/contribution_system.j2   — step 1 system prompt
#   templates/contribution_user.j2     — step 1 user prompt
#   templates/query_gen_system.j2      — step 2 system prompt
#   templates/query_gen_user.j2        — step 2 user prompt
# ==============================================================================

_TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
_jinja_env = Environment(loader=FileSystemLoader(_TEMPLATES_DIR), keep_trailing_newline=True)


def render_prompt(template_name: str, **kwargs) -> str:
    return _jinja_env.get_template(template_name).render(**kwargs)


def call_llm_json(full_prompt: str, system_prompt: str, model_name: str) -> Dict:
    combined_prompt = f"{system_prompt}\n\n{full_prompt}"
    print(f"\n[LLM Call] Model: {model_name}")
    print(f"System Prompt: {system_prompt[:100]}...")
    print(f"User Prompt: {full_prompt[:100]}...")

    response = prompt_openai_client(combined_prompt, engine=model_name)
    return extract_json_choice(response)


def generate_queries_custom(idea_text: str, model_name: str, n_contributions: int = 5, n_queries: int = 3) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Executes the query generation logic using the prompts defined above.
    Two-step approach: first extract contributions, then generate queries for all contributions at once.
    Returns a tuple of (queries_dict, contributions_dict).
    """
    print(f"\n--- Step 1: Extracting Contributions ---")

    user_prompt_1 = render_prompt("contribution_user.jinja2", idea_text=idea_text, n_contributions=n_contributions)
    system_prompt_1 = render_prompt("contribution_system.jinja2")
    contributions = call_llm_json(user_prompt_1, system_prompt_1, model_name)

    if not contributions:
        print("No contributions extracted (JSON parse failed or empty response).")
        return {}, {}

    print("\nExtracted Contributions:")
    print(json.dumps(contributions, indent=2))

    print(f"\n--- Step 2: Generating Queries for All Contributions ---")

    # Build a summary of all contributions
    contributions_lines = []
    for dimension, statements in contributions.items():
        if not isinstance(statements, list) or not statements:
            continue
        contributions_lines.append(f"{dimension}:")
        for s in statements:
            contributions_lines.append(f"  - {s}")

    contributions_summary = "\n".join(contributions_lines)
    print(f"\nContributions Summary:\n{contributions_summary}")

    user_prompt_2 = render_prompt("query_gen_user.jinja2", idea_text=idea_text, contributions_summary=contributions_summary)
    system_prompt_2 = render_prompt("query_gen_system.jinja2", n_queries=n_queries)

    response = call_llm_json(user_prompt_2, system_prompt_2, model_name)

    if response and 'queries' in response:
        queries = response['queries']
        print(f"\nGenerated {len(queries)} queries:")
        for q in queries:
            print(f"  - {q}")
        return {"all_contributions": queries}, contributions
    else:
        print("Failed to generate queries.")
        return {}, contributions


def print_summary(idea_text: str, queries: Dict[str, List[str]], contributions: Dict[str, List[str]], mode: str):
    """
    Prints a highly readable summary of the idea, contributions, and generated queries.
    """
    print("\n")
    print("=" * 80)
    print("                           SUMMARY")
    print("=" * 80)

    print("\n📄 IDEA TEXT:")
    print("-" * 80)
    # Word wrap the idea text for better readability
    words = idea_text.split()
    line = ""
    for word in words:
        if len(line) + len(word) + 1 <= 78:
            line = f"{line} {word}" if line else word
        else:
            print(f"  {line}")
            line = word
    if line:
        print(f"  {line}")

    # Display extracted contributions (only for two-step mode)
    if contributions:
        print("\n" + "-" * 80)
        print("\n💡 EXTRACTED CONTRIBUTIONS:")
        print("-" * 80)

        total_statements = 0
        for dimension, statements in contributions.items():
            if not isinstance(statements, list) or not statements:
                continue
            print(f"\n  🏷️  {dimension}")
            for i, statement in enumerate(statements, 1):
                # Word wrap long contribution statements
                wrapped_lines = []
                words = statement.split()
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 <= 70:
                        current_line = f"{current_line} {word}" if current_line else word
                    else:
                        wrapped_lines.append(current_line)
                        current_line = word
                if current_line:
                    wrapped_lines.append(current_line)

                # Print first line with number
                if wrapped_lines:
                    print(f"      {i}. {wrapped_lines[0]}")
                    # Print continuation lines with proper indentation
                    for continuation in wrapped_lines[1:]:
                        print(f"         {continuation}")
                total_statements += 1

        print("\n" + "-" * 80)
        print(f"  📊 Total: {len([d for d, s in contributions.items() if isinstance(s, list) and s])} dimension(s), {total_statements} contribution statement(s)")

    print("\n" + "-" * 80)
    print(f"\n🔍 GENERATED QUERIES (Mode: {mode}):")
    print("-" * 80)

    if not queries:
        print("  No queries were generated.")
    else:
        total_queries = 0
        for dimension, query_list in queries.items():
            print(f"\n  📌 {dimension.upper()}")
            for i, query in enumerate(query_list, 1):
                print(f"     {i}. {query}")
                total_queries += 1

        print("\n" + "-" * 80)
        print(f"  📊 Total: {len(queries)} dimension(s), {total_queries} query/queries")

    print("\n" + "=" * 80)


async def run_search_for_queries(queries_dict: Dict[str, List[str]], use_semantic_scholar: bool = False):
    print(f"\n================================================================================")
    print(f"TESTING SEARCH RESULTS (Top 3 per query)")
    print(f"================================================================================")

    all_queries = []
    for q_list in queries_dict.values():
        all_queries.extend(q_list)

    # Deduplicate
    unique_queries = list(set(all_queries))
    print(f"Total unique queries: {len(unique_queries)}")

    for query in unique_queries:
        print(f"\nQuery: '{query}'")
        try:
            if use_semantic_scholar:
                results = await asyncio.to_thread(SEMANTIC_API.search_papers, query, max_results=3)
                # Results is a dict {paperId: data}
                papers = list(results.values())
            else:
                results = await asyncio.to_thread(search_papers, query)
                papers = results.get("doc_collection", {}).get("documents", [])[:3]

            if not papers:
                print("  No results found.")
                continue

            for i, p in enumerate(papers):
                title = p.get('title', 'Unknown')
                year = p.get('year', 'N/A')
                venue = p.get('venue', 'N/A')
                print(f"  [{i + 1}] {title} ({venue}, {year})")
        except Exception as e:
            print(f"  Error during search: {e}")


def read_inputs(input_path: str) -> Dict[str, Any]:
    print(f"Reading inputs from {input_path}")
    with open(input_path, 'r') as f:
        return yaml.safe_load(f)


# default_abstract = """
# A hallmark of human innovation is recombination -- the creation of novel ideas by integrating elements from existing concepts and mechanisms. In this work, we introduce CHIMERA, a large-scale Knowledge Base (KB) of over 28K recombination examples automatically mined from the scientific literature. CHIMERA enables large-scale empirical analysis of how scientists recombine concepts and draw inspiration from different areas, and enables training models that propose novel, cross-disciplinary research directions. To construct this KB, we define a new information extraction task: identifying recombination instances in scientific abstracts. We curate a high-quality, expert-annotated dataset and use it to fine-tune a large language model, which we apply to a broad corpus of AI papers. We showcase the utility of CHIMERA through two applications. First, we analyze patterns of recombination across AI subfields. Second, we train a scientific hypothesis generation model using the KB, showing that it can propose novel research directions that researchers rate as inspiring.
# """

# default_abstract = """
# We introduce Debate Speech Evaluation as a novel and challenging benchmark for assessing LLM judges. Evaluating debate speeches requires a deep understanding of the speech at multiple levels, including argument strength and relevance, the coherence and organization of the speech, the appropriateness of its style and tone, and so on. This task involves a unique set of cognitive abilities that previously received limited attention in systematic LLM benchmarking. To explore such skills, we leverage a dataset of over 600 meticulously annotated debate speeches and present the first in-depth analysis of how state-of-the-art LLMs compare to human judges on this task. Our findings reveal a nuanced picture: while larger models can approximate individual human judgments in some respects, they differ substantially in their overall judgment behavior. We also investigate the ability of frontier LLMs to generate persuasive, opinionated speeches, showing that models may perform at a human level on this task.
# """


# default_abstract = """
# Understanding the impact of scientific publications is crucial for identifying breakthroughs and guiding future research. Traditional metrics based on citation counts often miss the nuanced ways a paper contributes to its field. In this work, we propose a new task: generating nuanced, expressive, and time-aware impact summaries that capture both praise (confirmation citations) and critique (correction citations) through the evolution of fine-grained citation intents. We introduce an evaluation framework tailored to this task, showing moderate to strong human correlation on subjective metrics such as insightfulness. Expert feedback from professors reveals a strong interest in these summaries and suggests future improvements.
# """

default_abstract = """
Despite the surge of interest in autonomous scientific discovery (ASD) of software artifacts (e.g., improved ML algorithms), current ASD systems face two key limitations: (1) they largely explore variants of existing codebases or similarly constrained design spaces, and (2) they produce large volumes of research artifacts (such as automatically generated papers and code) that are typically evaluated using conference-style paper review with limited evaluation of code. In this work we introduce CodeScientist, a novel ASD system that frames ideation and experiment construction as a form of genetic search jointly over combinations of research articles and codeblocks defining common actions in a domain (like prompting a language model). We use this paradigm to conduct hundreds of automated experiments on machine-generated ideas broadly in the domain of agents and virtual environments, with the system returning 19 discoveries, 6 of which were judged as being both at least minimally sound and incrementally novel after a multi-faceted evaluation beyond that typically conducted in prior work, including external (conference-style) review, code review, and replication attempts. Moreover, the discoveries span new tasks, agents, metrics, and data, suggesting a qualitative shift from benchmark optimization to broader discoveries.
"""


async def main():
    parser = argparse.ArgumentParser(description="Playground for designing retrieval query prompts.")
    parser.add_argument('--abstract', type=str, default=default_abstract,
                        help="Provide an idea/abstract text directly, bypassing input file lookup.")
    parser.add_argument('--input-file', type=str, default='benchmark_data/benchmark_instances/20260219_152105/benchmark_instances.yaml',
                        help="Path to input YAML file (ignored when --abstract is provided)")
    parser.add_argument('--problem-id', type=str, default=1, help="Problem ID to test (e.g., '0'). Defaults to first.")
    parser.add_argument('--idea-id', type=str, default=0, help="Idea ID to test (e.g., '0'). Defaults to first.")
    parser.add_argument('--llm-engine', type=str, default="claude-opus-4-6", help="LLM engine to use.")
    parser.add_argument('--run-search', action='store_true', help="Run search for generated queries to verify quality.")
    parser.add_argument('--use-semantic-scholar', action='store_true',
                        help="Use Semantic Scholar API instead of default.")
    parser.add_argument('--n-contributions', type=int, default=3,
                        help="Maximum number of contributions to generate.")
    parser.add_argument('--n-queries', type=int, default=5,
                        help="Number of queries to generate per contribution.")
    args = parser.parse_args()

    # --abstract bypasses file/problem/idea lookup entirely
    if args.abstract:
        idea_text = args.abstract
        print(f"\n================================================================================")
        print(f"IDEA (provided directly via --abstract)")
        print(f"================================================================================")
        print(f"Idea: {idea_text}")
        print(f"================================================================================")
    else:
        # Resolve input path
        input_path = args.input_file
        if not os.path.exists(input_path):
            # Try relative to this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            input_path = os.path.join(script_dir, args.input_file)

        if not os.path.exists(input_path):
            LOGGER.error(f"Input file not found: {args.input_file}")
            return

        inputs = read_inputs(input_path)

        # Select Problem
        problem_id = args.problem_id

        if problem_id not in inputs:
            LOGGER.error(f"Problem ID {problem_id} not found in inputs.")
            return

        problem_data = inputs[problem_id]
        ideas = problem_data.get('ideas', {})

        # Select Idea
        idea_id = args.idea_id

        if idea_id not in ideas:
            LOGGER.error(f"Idea ID {idea_id} not found in problem {problem_id}.")
            return

        idea_text = ideas[idea_id]
        context = problem_data.get('context', '')

        print(f"\n================================================================================")
        print(f"CONTEXT & IDEA")
        print(f"================================================================================")
        print(f"Context: {context}")
        print(f"-" * 80)
        print(f"Idea: {idea_text}")
        print(f"================================================================================")

    queries, contributions = generate_queries_custom(idea_text, args.llm_engine, args.n_contributions, args.n_queries)

    if args.run_search:
        await run_search_for_queries(queries, args.use_semantic_scholar)

    print_summary(idea_text, queries, contributions, "two-step")


if __name__ == "__main__":
    asyncio.run(main())