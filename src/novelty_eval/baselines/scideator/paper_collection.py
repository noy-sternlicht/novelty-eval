
import asyncio
import os

from .retrieval.query_based_retrieval import get_query_based_papers_helper
from .ranking.embedding import get_embeddings_ideapapers
from .retrieval.input_collection import get_papers_similar_to_input_papers
from .ranking.embedding_ranker import embeddingFiltering
from .ranking.llm_reranker import rank_gpt_filter
from .prompts_pipeline import prompt_RankGPT_IdeaFacets
from .._shared.llm_adapters import SharedLLMClient


async def get_priority_facets(idea):

    priority_prompt = prompt_RankGPT_IdeaFacets(idea)
    model = os.getenv("DEFAULT_MODEL", "gpt-5.1")
    client = SharedLLMClient(model=model, stage="scideator_priority_facets")
    output, _ = client.chat(
        model=model,
        messages=(
            [{"role": "system", "content": priority_prompt}]
            if isinstance(priority_prompt, str)
            else priority_prompt
        ),
        temperature=float(os.getenv("DEFAULT_TEMPERATURE", 0)),
        return_history=True,
    )
    return output


async def get_query_based_papers(idea, seed_papers, max_papers_specter=100):

    query_retrieval_trace = await get_query_based_papers_helper(idea)

    snippet_papers_temp = query_retrieval_trace["snippet_papers"]
    snippet_papers = {}
    for k, v in snippet_papers_temp.items():
        v["source"] = "snippet"
        snippet_papers[k] = v

    keyword_papers_temp = query_retrieval_trace["keyword_papers"]
    keyword_papers = {}
    for k, v in keyword_papers_temp.items():
        if snippet_papers.get(k, None) is not None:
            snippet_papers[k]["source"] = "keyword+title+snippet"
        else:
            v["source"] = "keyword+title"
            keyword_papers[k] = v

    idea_papers = snippet_papers.copy()
    idea_papers.update(keyword_papers)

    input_papers_temp = seed_papers.copy()
    input_papers = {}
    corpusIds = []
    for k, v in input_papers_temp.items():
        v["source"] = "input"
        input_papers[k] = v
        corpusIds.append(v["corpusId"])

    input_paper_similar_temp = await get_papers_similar_to_input_papers(corpusIds)
    input_paper_similar = {}
    for k, v in input_paper_similar_temp.items():
        v["source"] = "similar_to_input"
        input_paper_similar[k] = v

    input_papers = await get_embeddings_ideapapers(input_papers)
    input_paper_similar = await get_embeddings_ideapapers(input_paper_similar)
    input_papers.update(input_paper_similar)
    input_papers.update(idea_papers)

    most_similar_documents = await embeddingFiltering(
        idea, input_papers, max_papers_specter
    )

    query_retrieval_trace.update({"embedding_ranked": most_similar_documents})

    return query_retrieval_trace


async def get_most_relevant_papers(idea, seed_papers, max_papers_specter=100):

    priority_facets_task = get_priority_facets(idea)
    novelty_checker_task = get_query_based_papers(idea, seed_papers, max_papers_specter)

    idea_priority_facets, query_retrieval_trace = await asyncio.gather(
        priority_facets_task, novelty_checker_task
    )

    most_similar_documents_filtered = query_retrieval_trace["embedding_ranked"].dropna(
        subset=["title", "abstract"]
    )

    similar_documents = most_similar_documents_filtered.copy()

    hits = [
        {"content": f"Title: {row['title']}. Content: {row['abstract']}"}
        for _, row in similar_documents.iterrows()
    ]

    item = {"query": idea, "hits": hits}
    similar_documents["content"] = [i["content"] for i in item["hits"]]

    await rank_gpt_filter(
        item,
        similar_documents,
        os.getenv("RANKGPT_MODEL"),
        os.getenv("RANKGPT_VARIANT"),
        idea_priority_facets,
    )

    similar_documents.sort_values(
        f'rankGPT_{os.getenv("RANKGPT_VARIANT")}', inplace=True
    )

    query_retrieval_trace.update(
        {
            "idea_priority_facets": idea_priority_facets,
            "most_relevant_papers": similar_documents,
        }
    )
    return query_retrieval_trace
