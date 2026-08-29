
import re
import os
import pathlib
import asyncio

from .prompts import (
    prompt_NoveltyChecker_allowsIncrementalNovelty,
    prompt_NoveltyChecker_allowsIncrementalNovelty_lessRelaxed,
)
from .._shared.llm_adapters import SharedLLMClient


_INCONTEXT_DIR = pathlib.Path(__file__).parent / "incontext_examples"


def clean_text(text: str) -> str:
    return text.strip(" \n#:][]")


def parse_output(text: str, delim_class: str, delim_review: str):
    category_part = text.split(delim_review)[0]
    category = category_part.split(delim_class)[-1].strip()
    review = text.split(delim_review)[1].strip()

    for punc in ["*", ":", "\n"]:
        category = re.sub(f"\\{punc}", "", category)
        review = re.sub(f"\\{punc}", "", review)

    return clean_text(category).lower(), clean_text(review)


def get_prompt_and_parsing_rules(idea, most_relevant_papers, incontext_example):

    delim_class = "Class:"
    delim_review = "Review:"
    if os.getenv("NOVELTY_CHECK_PROMPT") == "relaxed":
        prompt = prompt_NoveltyChecker_allowsIncrementalNovelty(
            idea, most_relevant_papers, incontext_example
        )
    elif os.getenv("NOVELTY_CHECK_PROMPT") == "less-relaxed":
        prompt = prompt_NoveltyChecker_allowsIncrementalNovelty_lessRelaxed(
            idea, most_relevant_papers, incontext_example
        )
    else:
        raise ValueError("Invalid novelty checker prompt style.")
    return prompt, delim_class, delim_review


async def get_review(idea, most_relevant_papers, incontext_example_path=None):

    model = os.getenv("NOVELTY_CHECK_MODEL", "gpt-5.1")
    temperature = float(os.getenv("NOVELTY_CHECK_TEMPERATURE", 0))

    temperature = 0
    if incontext_example_path is not None:
        incontext_example = open(incontext_example_path, "r")

    if os.getenv("NOVELTY_CHECK_EXAMPLES") == "less-relaxed":
        incontext_example = open(_INCONTEXT_DIR / "less-relaxed.json", "r")
    elif os.getenv("NOVELTY_CHECK_EXAMPLES") == "relaxed":
        incontext_example = open(_INCONTEXT_DIR / "relaxed.json", "r")
    else:
        raise ValueError("Invalid incontext example file path.")

    prompt, delim_class, delim_review = get_prompt_and_parsing_rules(
        idea, most_relevant_papers, incontext_example
    )

    client = SharedLLMClient(model=model, stage="scideator_verdict")

    if model in ["o3-mini", "o1"]:
        output_text, _ = await client.a_chat(
            model=model, messages=prompt, return_history=True,
        )
    else:
        output_text, _ = await client.a_chat(
            model=model, messages=prompt, temperature=temperature, return_history=True,
        )

    category, review = parse_output(output_text, delim_class, delim_review)
    if category is None or review is None:
        return None, None, output_text
    category = category.strip("-").strip(" ")
    review = review.strip("-").strip(" ")
    return category, review, output_text
