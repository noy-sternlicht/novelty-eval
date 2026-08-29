import asyncio
from typing import List

from ..._shared.s2_client import (
    papers_from_recommendation_api_allCs,
    papers_from_recommendation_api_recent,
)


async def get_papers_similar_to_input_papers(corpusIds: List[str]):

    if not corpusIds:
        return {}

    all_papers = []
    for corpus_id in corpusIds:
        papers = await papers_from_recommendation_api_allCs(corpus_id)
        if papers.get("recommendedPapers"):
            all_papers.extend(papers["recommendedPapers"])
        papers = await papers_from_recommendation_api_recent(corpus_id)
        if papers.get("recommendedPapers"):
            all_papers.extend(papers["recommendedPapers"])

    all_papers = {
        str(paper["corpusId"]): {k: v for k, v in paper.items()} for paper in all_papers
    }

    return all_papers
