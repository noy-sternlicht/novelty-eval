import os

from .._shared.llm_adapters import SharedLLMClient
from .._shared.s2_client import papers_from_search_api
from .prompt import novelty_system_msg, novelty_prompt
from .utils import extract_json_between_markers


async def get_review(
    idea,
    input_papers=None,
    max_num_iterations: int = 10,
    use_retrieval: bool = True,
    *,
    model: str | None = None,
    effort: str = "medium",
    cutoff_date: str | None = None,
    exclude_titles: list | None = None,
):
    """Returns (novel: bool, msg_history: list, iteration_metadata: dict).

    `iteration_metadata` is a per-round trace: {round_idx: {paper, query, papers, category}}.
    Callers should serialize this verbatim — it's the baseline's full trace.
    """

    model = model or os.getenv("NOVELTY_CHECK_MODEL") or "gpt-5.1"
    temperature = float(os.getenv("NOVELTY_CHECK_TEMPERATURE", 0))

    client = SharedLLMClient(model=model, effort=effort, stage="ai_scientist_novelty")

    novel = False
    msg_history: list = []
    papers_str = ""
    iteration_metadata: dict = {}

    for j in range(max_num_iterations):
        iteration_metadata[j] = {"paper": [], "query": [], "category": False}
        try:
            text, msg_history = await client.a_chat(
                messages=novelty_prompt.format(
                    current_round=j + 1,
                    num_rounds=max_num_iterations,
                    idea=idea,
                    last_query_results=papers_str,
                ),
                model=model,
                temperature=temperature,
                system_message=novelty_system_msg.format(num_rounds=max_num_iterations),
                msg_history=msg_history,
                return_history=True,
            )
            if "decision made: novel" in text.lower():
                novel = True
                iteration_metadata[j]["category"] = novel
                iteration_metadata[j]["raw_response"] = text
                break
            if "decision made: not novel" in text.lower():
                iteration_metadata[j]["raw_response"] = text
                break

            # PARSE OUTPUT
            json_output = extract_json_between_markers(text)
            assert json_output is not None, "Failed to extract JSON from LLM output"

            # SEARCH FOR PAPERS
            query = json_output["Query"]
            iteration_metadata[j]["query"] = query
            iteration_metadata[j]["raw_response"] = text
            papers = await papers_from_search_api(
                query, cutoff_date=cutoff_date, exclude_titles=exclude_titles,
            )
            iteration_metadata[j]["papers"] = papers

            if papers["data"] is None:
                papers_str = "No papers found."

            paper_strings = []

            # Include input papers if provided (this part is changed from original code)
            if input_papers is not None:
                for i, row in input_papers.iterrows():
                    paper_strings.append(
                        """{i}: {title}. {authors}. {venue}, {year}.\nNumber of citations: {cites}\nAbstract: {abstract}""".format(
                            i=i,
                            title=row["title"],
                            authors=row.get("authors", "unknown"),
                            venue=row.get("venue", "unknown"),
                            year=row.get("year", "unknown"),
                            cites=row.get("citationCount", "unknown"),
                            abstract=row["abstract"],
                        )
                    )

            if use_retrieval:
                for i, paper in enumerate(papers.get("data") or []):
                    paper_strings.append(
                        """{i}: {title}. {authors}. {venue}, {year}.\nNumber of citations: {cites}\nAbstract: {abstract}""".format(
                            i=i,
                            title=paper.get("title", ""),
                            authors=paper.get("authors", ""),
                            venue=paper.get("venue", ""),
                            year=paper.get("year", ""),
                            cites=paper.get("citationCount", ""),
                            abstract=paper.get("abstract", ""),
                        )
                    )

            papers_str = "\n\n".join(paper_strings)

        except Exception as e:
            iteration_metadata[j]["error"] = f"{type(e).__name__}: {e}"
            print(f"AI-Scientist round {j} error: {e}")
            continue

    return novel, msg_history, iteration_metadata
