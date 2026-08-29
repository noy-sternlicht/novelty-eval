#!/usr/bin/env python3
import argparse
import os
import sys
import json
import textwrap
from jinja2 import Environment, FileSystemLoader

# Add src to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# The utils module automatically loads API keys from secrets.toml upon import
from utils import prompt_openai_client, extract_json_choice

# Initialize Jinja2 environment
current_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(os.path.dirname(current_dir), 'templates')
env = Environment(loader=FileSystemLoader(templates_dir))

# DEFAULT_ABSTRACT = """
# Large language models (LLMs) have demonstrated remarkable capabilities in chain of thought (CoT) reasoning. However, the current LLM reasoning paradigm initiates thinking only after the entire input is available, which introduces unnecessary latency and weakens attention to earlier information in dynamic scenarios. Inspired by human cognition of thinking while reading, we first design a **streaming thinking** paradigm for LLMs, where reasoning unfolds in the order of input and further adjusts its depth once reading is complete. We instantiate this paradigm with *StreamingThinker*, a framework that enables LLMs to think while reading through the integration of streaming CoT generation, streaming-constraint training, and streaming parallel inference. Specifically, StreamingThinker employs streaming reasoning units with quality control for CoT generation, enforces order-preserving reasoning through streaming attention masks and position encoding, and leverages parallel KV caches that decouple input encoding from reasoning generation, thereby ensuring alignment and enabling true concurrency.
# """

# DEFAULT_ABSTRACT = """
# Large Reasoning Models (LRMs) have the ability to self-correct even when they make mistakes in their reasoning paths. However, our study reveals that when the reasoning process starts with a short but poor beginning, it becomes difficult for the model to recover. We refer to this phenomenon as the *"Prefix Dominance Trap"*. This phenomenon indicates that the self-correction ability of LRMs is fragile and can be easily derailed by a poor start. This fragility motivates us to **look beyond internal self-correction**. Inspired by psychological findings that peer interaction can promote correction ability without negatively impacting already accurate individuals, we propose **Learning from Peers** (LeaP) to address this phenomenon. LeaP enables reasoning paths to periodically (every T tokens) summarize and share intermediate reasoning via a routing mechanism, thereby incorporating peer insights. For smaller models that may inefficiently follow summarization and reflection instructions, we introduce fine-tuned **LeaP-T** models. In-depth analysis reveals that LeaP provides robust error correction through timely peer insights and exhibits strong error tolerance.
# """

DEFAULT_ABSTRACT = """
  A hallmark of human innovation is recombination -- the creation of novel ideas
  by integrating elements from existing concepts and mechanisms. In this work,
  we introduce CHIMERA, a large-scale Knowledge Base (KB) of over 28K
  recombination examples automatically mined from the scientific literature.
  CHIMERA enables large-scale empirical analysis of how scientists recombine
  concepts and draw inspiration from different areas, and enables training
  models that propose novel, cross-disciplinary research directions. To
  construct this KB, we define a new information extraction task: identifying
  recombination instances in scientific abstracts. We curate a high-quality,
  expert-annotated dataset and use it to fine-tune a large language model, which
  we apply to a broad corpus of AI papers. We showcase the utility of CHIMERA
  through two applications. First, we analyze patterns of recombination across
  AI subfields. Second, we train a scientific hypothesis generation model using
  the KB to propose novel research directions.

"""


# DEFAULT_ABSTRACT = """
#   We introduce Debate Speech Evaluation as a novel and challenging benchmark for
#   assessing LLM judges. Evaluating debate speeches requires a deep understanding
#   of the speech at multiple levels, including argument strength and relevance,
#   the coherence and organization of the speech, the appropriateness of its style
#   and tone, and so on. This task involves a unique set of cognitive abilities
#   that previously received limited attention in systematic LLM benchmarking. To
#   explore such skills, we leverage a dataset of over 600 meticulously annotated
#   debate speeches and present the first in-depth analysis of how state-of-the-
#   art LLMs compare to human judges on this task. We also investigate the ability
#   of frontier LLMs to generate persuasive, opinionated speeches.
# """

def get_extraction_prompt(abstract):
    template = env.get_template('extract_research_plan.jinja2')
    return template.render(abstract=abstract)

def run_extraction(abstract, model_name):
    print(f"\n{'='*80}")
    print(f"--- Running Extraction with {model_name} ---")
    print(f"{'='*80}")
    prompt = get_extraction_prompt(abstract)
    
    try:
        # prompt_openai_client is synchronous
        # It handles both OpenAI and Anthropic models based on the model name
        response = prompt_openai_client(prompt, engine=model_name)
        json_response = extract_json_choice(response)
        
        if json_response:
            # Check if the response has the expected 'idea' key
            idea_content = json_response.get('idea', json_response)
            
            print(f"✅ Extracted Research Plan:")
            if isinstance(idea_content, dict):
                for key, value in idea_content.items():
                    print(f"\n🔹 {key.capitalize()}:")
                    # Wrap text for better readability
                    wrapped_text = textwrap.fill(str(value), width=80, initial_indent="  ", subsequent_indent="  ")
                    print(wrapped_text)
            else:
                print(json.dumps(idea_content, indent=2))
            print(f"\n{'='*80}")
        else:
            print(f"❌ Raw response from {model_name} (JSON extraction failed):")
            print(response)
            
    except Exception as e:
        print(f"Error running model {model_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Playground for extract_research_plan.jinja2")
    parser.add_argument('--abstract', type=str, default=DEFAULT_ABSTRACT, help="Abstract text to process")
    parser.add_argument('--gpt-model', type=str, default="gpt-5.4", help="GPT model name")
    parser.add_argument('--claude-model', type=str, default="claude-opus-4-6", help="Claude model name")
    
    args = parser.parse_args()
    
    print(f"\n📄 Abstract:\n{'-'*80}")
    print(textwrap.fill(args.abstract, width=80))
    print(f"{'-'*80}")
    
    # Run for GPT
    run_extraction(args.abstract, args.gpt_model)
    
    # Run for Claude
    run_extraction(args.abstract, args.claude_model)

if __name__ == "__main__":
    main()
