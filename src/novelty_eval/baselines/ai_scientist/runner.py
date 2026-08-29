"""AI Scientist novelty checker, plugged into the BaselineRunner contract.

Pipeline (per the paper / Sakana AI-Scientist repo):
  - up to N iterative rounds (default 10)
  - each round: LLM proposes a search query → S2 retrieves top results → LLM
    decides "novel" / "not novel" / continue
  - retrieval is the baseline's OWN (uses Semantic Scholar via _shared/s2_client),
    NOT the in-house retrieval_cache (per Si-et-al-style comparison fairness:
    each baseline retrieves as its paper describes).
"""
from __future__ import annotations

from typing import Any, Optional

from ..base import BaselineRunner, PointwiseResult
from .check_novelty import get_review

# The literal markers `check_novelty` breaks the round loop on.
_DECISION_MARKERS = ("decision made: novel", "decision made: not novel")


def _reached_a_verdict(iteration_metadata: dict) -> bool:
    """Did any round actually state a verdict?

    `get_review` initialises `novel = False` and returns it whether the model
    said "not novel" or simply ran out of rounds, so the two are otherwise
    indistinguishable. running out of rounds is not a judgement, and scoring it as "not novel"
    would be automatically correct on every NEGATIVE instance.
    """
    for meta in (iteration_metadata or {}).values():
        text = (meta.get("raw_response") or "").lower()
        if any(marker in text for marker in _DECISION_MARKERS):
            return True
    return False


class AIScientistRunner(BaselineRunner):
    name = "ai_scientist"
    paper = "Lu et al., The AI Scientist (Sakana AI, 2024) — novelty stage"
    supports = {"pointwise"}

    async def run_pointwise(
        self,
        instance_id: str,
        idea: str,
        *,
        llm_engine: str,
        retrieval_cache_entry: Optional[dict] = None,  # ignored — baseline retrieves itself
        topic: Optional[str] = None,
        max_num_iterations: int = 10,
        use_retrieval: bool = True,
        effort: str = "medium",
        cutoff_date: Optional[str] = None,
        exclude_titles: Optional[list] = None,
        **_: Any,
    ) -> Optional[PointwiseResult]:
        novel, msg_history, iteration_metadata = await get_review(
            idea,
            input_papers=None,
            max_num_iterations=max_num_iterations,
            use_retrieval=use_retrieval,
            model=llm_engine,
            effort=effort,
            cutoff_date=cutoff_date,
            exclude_titles=exclude_titles,
        )

        # Build a rich trace — everything the baseline saw + decided.
        rounds = []
        total_papers_found = 0
        for round_idx in sorted(iteration_metadata.keys()):
            meta = iteration_metadata[round_idx]
            papers = meta.get("papers") or {}
            paper_list = papers.get("data") or [] if isinstance(papers, dict) else []
            papers_clean = [
                {
                    "title": p.get("title", ""),
                    "authors": p.get("authors", ""),
                    "year": p.get("year", ""),
                    "venue": p.get("venue", ""),
                    "abstract": p.get("abstract", ""),
                    "paperId": p.get("paperId", ""),
                    "citationCount": p.get("citationCount", 0),
                }
                for p in paper_list
                if isinstance(p, dict)
            ]
            total_papers_found += len(papers_clean)
            rounds.append({
                "round": round_idx + 1,
                "query": meta.get("query", ""),
                "papers": papers_clean,
                "decided": meta.get("category", False),
                "raw_response": meta.get("raw_response", ""),
                "error": meta.get("error"),
            })

        reached_verdict = _reached_a_verdict(iteration_metadata)

        trace = {
            "category": ("novel" if novel else "not novel") if reached_verdict else "undecided",
            "reached_verdict": reached_verdict,
            "exhausted_rounds": not reached_verdict,
            # Which model ran which stage, so a run directory answers that on
            # its own instead of requiring the source and the env to be read.
            "models": {"novelty_check": llm_engine},
            "n_rounds": len(rounds),
            "total_papers_found": total_papers_found,
            "rounds": rounds,
            "msg_history": msg_history,
            "model": llm_engine,
            "max_num_iterations": max_num_iterations,
            "use_retrieval": use_retrieval,
            "cutoff_date": cutoff_date,
            "exclude_titles": exclude_titles or [],
        }

        return PointwiseResult(
            # Running out of rounds is not a verdict. Returning None routes the
            # instance through the adapter's no-prediction path, so it is left
            # out of scores.json and injected as wrong, rather than scoring as a
            # confident "not novel" that happens to be right half the time.
            prediction=(1 if novel else 0) if reached_verdict else None,
            trace=trace,
        )
