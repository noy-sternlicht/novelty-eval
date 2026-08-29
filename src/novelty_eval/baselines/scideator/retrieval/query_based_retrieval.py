import re
import os
import ast
import asyncio

import nest_asyncio

from ..._shared.llm_adapters import SharedLLMClient
from ..._shared.s2_client import (
    papers_from_search_api_with_snippet_mode as papers_from_search_api,
    get_paper_data,
)
from ..prompts_pipeline import prompt_PaperRetrieval_Keywords
from ..ranking.embedding import get_embeddings_ideapapers

nest_asyncio.apply()


async def get_keywords(input_string):

    combined_prompt = prompt_PaperRetrieval_Keywords(input_string)

    def parse_keyword_response(output):
        keyword_match = re.search(r"<keywords>(.*?)</keywords>", output, re.DOTALL)
        title_match = re.search(r"<titles>(.*?)</titles>", output, re.DOTALL)

        keywords = []
        titles = []

        if keyword_match:
            keyword_string = keyword_match.group(1)
            try:
                keywords = ast.literal_eval(keyword_string.strip())
            except (ValueError, SyntaxError):
                pass

        if title_match:
            title_string = title_match.group(1)
            try:
                titles = ast.literal_eval(title_string.strip())
            except (ValueError, SyntaxError):
                pass

        return keywords, titles

    model = os.getenv("DEFAULT_MODEL", "gpt-5.1")
    client = SharedLLMClient(model=model, stage="scideator_keywords")
    output, _ = await client.a_chat(
        model=model,
        messages=combined_prompt,
        temperature=float(os.getenv("DEFAULT_TEMPERATURE", 0)),
        return_history=True,
    )
    keywords, titles = parse_keyword_response(output)

    return keywords, titles


async def fetch_papers_for_query(query, search_type, limit):
    output = await papers_from_search_api(query, search_type=search_type, limit=limit)
    if output is None:
        return []
    if isinstance(output, list):
        return output
    if isinstance(output, dict) and output.get("data") is not None:
        return output["data"]
    return []


# S2 allows 500 ids per /paper/batch request. Kept well below that so a failed
# chunk costs a bounded slice of the results rather than all of them.
_BATCH_CHUNK = 100


async def run_all_queries(corpus_ids):
    results = []
    for start in range(0, len(corpus_ids), _BATCH_CHUNK):
        chunk = corpus_ids[start:start + _BATCH_CHUNK]
        fetched = await get_paper_data(
            [f"CorpusId:{cid}" for cid in chunk], batch_wise=True,
        )
        if not fetched:
            # A failed chunk loses that slice; keep the positions so a caller
            # comparing requested against retrieved sees the shortfall.
            results.extend([None] * len(chunk))
            continue
        results.extend(fetched)
    return results


async def get_query_based_papers_helper(idea):
    keyword_papers = []
    snippet_papers = {}
    idea_keywords = []
    title_keywords = []
    search_type = os.getenv("QUERY_RETRIEVAL_METHOD")

    if search_type == "keyword+title" or search_type == "keyword+title+snippet":
        current_search_type = "keyword"
        idea_keywords, title_keywords = await get_keywords(input_string=idea)
        search_queries = idea_keywords + title_keywords

        results = await asyncio.gather(
            *[
                fetch_papers_for_query(query, current_search_type, limit=100)
                for query in search_queries
            ]
        )
        for result in results:
            if result:
                keyword_papers.extend(result)

    if search_type == "snippet" or search_type == "keyword+title+snippet":
        current_search_type = "snippet"
        snippet_papers_temp = await fetch_papers_for_query(
            idea, current_search_type, limit=100
        )
        corpus_IDs = []
        for paper in snippet_papers_temp:
            if paper and isinstance(paper, dict):
                paper_info = paper.get("paper", {})
                if paper_info and isinstance(paper_info, dict):
                    cid = paper_info.get("corpusId")
                    if cid:
                        corpus_IDs.append(cid)

        if corpus_IDs:
            snippet_papers = asyncio.run(run_all_queries(corpus_IDs))
            snippet_papers = [
                p for p in snippet_papers
                if p is not None and isinstance(p, dict) and p.get("corpusId")
            ]
        else:
            snippet_papers = []

    keyword_papers = [
        p for p in keyword_papers
        if p is not None and isinstance(p, dict) and p.get("corpusId")
    ]

    snippet_papers = await get_embeddings_ideapapers(snippet_papers)
    keyword_papers = await get_embeddings_ideapapers(keyword_papers)

    return {
        "search_type": search_type,
        "snippet_papers": snippet_papers,
        "keyword_papers": keyword_papers,
        "idea_keywords": idea_keywords,
        "title_keywords": title_keywords,
    }
