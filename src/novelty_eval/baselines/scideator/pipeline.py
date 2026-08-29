
import os
import asyncio

from tqdm import tqdm
import pandas as pd

from .check_novelty import get_review
from .paper_collection import get_most_relevant_papers
from .._shared.s2_client import get_paper_data


async def run_ideanoveltychecker(
    idea, use_retrieval=True, input_papers_ids=[], input_papers=None, ablation=False
):
    """
    idea: Idea string. Check preferred format in incontext examples.
    use_retrieval: If True, will find more papers similar to idea from S2.
    input_papers_ids: list of paperIds important for consideration in most relevant set.
    ablation: compute novelty with different retrieval-component combinations.
    """

    retrieval_trace = {
        "idea_keywords": [],
        "title_keywords": [],
        "idea_priority_facets": "",
        "most_relevant_papers": [],
        "embedding_ranked": [],
        "snippet_papers": [],
        "keyword_papers": [],
    }

    limit = int(os.getenv("NOVELTY_CHECK_TOPkPapers", 10))

    if len(input_papers_ids) != 0 and input_papers is None:
        input_papers = await get_paper_data(
            input_papers_ids, id_type="paper_id", batch_wise=True
        )
        input_papers = pd.DataFrame(input_papers)
    elif input_papers is not None:
        if isinstance(input_papers, list):
            input_papers = pd.DataFrame(input_papers)
    else:
        # Match upstream behaviour: empty DataFrame, not list — paper_collection
        # iterates with .items() which only works on dict/DataFrame.
        input_papers = pd.DataFrame([])

    if use_retrieval is True:
        retrieval_trace = await get_most_relevant_papers(
            idea, input_papers, max_papers_specter=100
        )

    retrieval_trace["input_most_relevant_papers"] = input_papers

    output = {
        "input": {
            "idea": idea,
            "input_papers_ids": input_papers_ids,
            "use_retrieval": use_retrieval,
        },
        "trace": retrieval_trace,
        "output": {},
    }

    most_relevant_papers = retrieval_trace["most_relevant_papers"].copy() if hasattr(retrieval_trace["most_relevant_papers"], "copy") else []

    if ablation:
        snippet_only_ablation = most_relevant_papers[
            most_relevant_papers["source"].apply(
                lambda x: True if "snippet" in x else False
            )
        ]
        keyword_papers_ablation = most_relevant_papers[
            most_relevant_papers["source"].apply(
                lambda x: True if "keyword" in x else False
            )
        ]

        experiments = [
            ("default", retrieval_trace["most_relevant_papers"][:limit]),
            ("snippet_only", snippet_only_ablation[:limit]),
            ("keyword_only", keyword_papers_ablation[:limit]),
            ("norankGPT", retrieval_trace["embedding_ranked"][:limit]),
            (
                "groundtruth_only",
                (
                    retrieval_trace["input_most_relevant_papers"][:limit]
                    if (len(retrieval_trace["input_most_relevant_papers"]) != 0)
                    is not None
                    else []
                ),
            ),
            (
                "groundtruth_and_retrieved",
                (
                    pd.concat(
                        [
                            retrieval_trace["input_most_relevant_papers"],
                            retrieval_trace["most_relevant_papers"],
                        ]
                    )[:limit]
                    if (len(retrieval_trace["input_most_relevant_papers"]) != 0)
                    and (len(retrieval_trace["most_relevant_papers"]) != 0)
                    else []
                ),
            ),
        ]
    else:
        if use_retrieval:
            experiments = [("default", retrieval_trace["most_relevant_papers"][:limit])]
        else:
            experiments = [
                (
                    "groundtruth_only",
                    (
                        retrieval_trace["input_most_relevant_papers"][:limit]
                        if retrieval_trace["input_most_relevant_papers"] is not None
                        else []
                    ),
                )
            ]

    for experiment_type, evaluation_papers in tqdm(
        experiments, total=len(experiments), desc="running_experiments..."
    ):

        if len(evaluation_papers) == 0:
            continue

        category, review, output_text = "", "", ""

        if len(evaluation_papers) != 0:
            category, review, output_text = await get_review(
                idea=idea,
                most_relevant_papers=evaluation_papers,
            )
        try:
            evaluation_papers.drop(["embedding"], axis=1, inplace=True)
        except Exception:
            pass

        output["output"][experiment_type] = {
            "evaluation_papers": evaluation_papers.to_dict(orient="records") if hasattr(evaluation_papers, "to_dict") else list(evaluation_papers),
            "category": category,
            "review": review,
            "output_novelty_text": output_text,
        }

    return output
