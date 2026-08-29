
from __future__ import annotations

import os
import threading
from typing import Any, Optional

from ..base import BaselineRunner, PointwiseResult
from .pipeline import run_ideanoveltychecker


def _category_to_prediction(category: str) -> Optional[int]:
    """Scideator emits 'novel' / 'not novel' (parsed from `Class: ...`).

    Returns None when the category is not recognisable at all, rather than
    falling back to 0. "Not novel" is the correct answer on every NEGATIVE
    instance, so defaulting to it would score an unreadable verdict as a
    confident correct answer on half the benchmark. None instead routes through
    the adapter's no-prediction path: the instance is left out of scores.json,
    and the ablation layer injects it as wrong — the treatment a failure is
    meant to get.
    """
    cat = (category or "").strip().rstrip("]").strip().lower()
    if cat == "novel":
        return 1
    if cat in ("not novel", "not_novel", "notnovel"):
        return 0
    # Wording variants that still clearly state a verdict, e.g. "incrementally
    # novel" (→ novel) or "not quite novel" (→ not novel).
    if "novel" in cat:
        return 0 if "not" in cat else 1
    return None


def _df_to_records(df_or_list: Any) -> list:
    """Some intermediate retrieval results are pandas DataFrames; serialize
    them defensively so the trace is JSON-safe.
    """
    if df_or_list is None:
        return []
    if hasattr(df_or_list, "to_dict"):
        try:
            df_or_list = df_or_list.drop(columns=["embedding"], errors="ignore")
        except Exception:
            pass
        try:
            return df_or_list.to_dict(orient="records")
        except Exception:
            return []
    if isinstance(df_or_list, dict):
        out = {}
        for k, v in df_or_list.items():
            vv = dict(v) if isinstance(v, dict) else v
            if isinstance(vv, dict):
                vv.pop("embedding", None)
            out[k] = vv
        return out
    if isinstance(df_or_list, list):
        return df_or_list
    return []


class ScideatorRunner(BaselineRunner):
    name = "scideator"
    paper = "Radensky et al., Scideator (2024) — Idea Novelty Checker"
    supports = {"pointwise"}

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._env_lock = threading.Lock()
        self._applied_env: Optional[dict] = None

    def _ensure_env(self, desired: dict) -> None:
        """Publish Scideator's settings to the environment exactly once.

        The vendored code reads its configuration through `os.getenv`, which is
        process-global. The original wrote those variables at the top of every
        call and restored them afterwards — safe when one instance runs at a
        time, but the adapter now runs instances concurrently, so two calls
        would interleave their writes and read each other's settings.

        Nothing here varies per instance: every value comes from `llm_engine`
        and `baseline_kwargs`, both fixed for a run. So they are applied on the
        first call and left alone, which removes the race rather than guarding
        it. A second, differing configuration in the same process cannot be
        expressed this way and is rejected instead of silently winning.
        """
        with self._env_lock:
            if self._applied_env == desired:
                return
            if self._applied_env is not None:
                changed = {
                    k: (self._applied_env.get(k), v)
                    for k, v in desired.items()
                    if self._applied_env.get(k) != v
                }
                raise RuntimeError(
                    "Scideator settings changed within a single process; its "
                    "vendored code reads them from the global environment, so "
                    f"they cannot vary per run. Changed: {changed}"
                )
            os.environ.update(desired)
            self._applied_env = dict(desired)

    async def run_pointwise(
        self,
        instance_id: str,
        idea: str,
        *,
        llm_engine: str,
        retrieval_cache_entry: Optional[dict] = None,  # not used — baseline retrieves itself
        topic: Optional[str] = None,
        use_retrieval: bool = True,
        ablation: bool = False,
        novelty_check_prompt: str = "relaxed",      # "relaxed" | "less-relaxed"
        novelty_check_examples: str = "relaxed",
        novelty_check_topk: int = 10,
        query_retrieval_method: str = "keyword+title+snippet",
        rankgpt_model: Optional[str] = None,
        rankgpt_effort: Optional[str] = None,       # defaults to `effort`, as the model does
        rankgpt_variant: str = "priority",          # "base" | "purpose" | "priority"
        effort: str = "medium",
        **_: Any,
    ) -> Optional[PointwiseResult]:

        # Scideator reads its configuration through os.getenv, so it has to be
        # published to the environment. None of it varies per instance — see
        # _ensure_env.
        self._ensure_env({
            "NOVELTY_CHECK_MODEL": llm_engine,
            "NOVELTY_CHECK_TEMPERATURE": "0",
            "NOVELTY_CHECK_PROMPT": novelty_check_prompt,
            "NOVELTY_CHECK_EXAMPLES": novelty_check_examples,
            "NOVELTY_CHECK_TOPkPapers": str(novelty_check_topk),
            "QUERY_RETRIEVAL_METHOD": query_retrieval_method,
            "DEFAULT_MODEL": llm_engine,
            "DEFAULT_TEMPERATURE": "0",
            "RANKGPT_MODEL": rankgpt_model or llm_engine,
            "RANKGPT_REASONING_EFFORT": rankgpt_effort or effort,
            "RANKGPT_VARIANT": rankgpt_variant,
            "BASELINE_REASONING_EFFORT": effort,
        })

        result = await run_ideanoveltychecker(
            idea=idea,
            use_retrieval=use_retrieval,
            input_papers_ids=[],
            input_papers=None,
            ablation=ablation,
        )

        # Pick the "default" experiment for prediction (matches upstream behaviour
        # when ablation=False).
        out_map = result.get("output", {})
        default_run = out_map.get("default") or next(iter(out_map.values()), {})
        category = default_run.get("category", "")
        review = default_run.get("review", "")
        prediction = _category_to_prediction(category)

        # Build a rich, JSON-safe trace with everything the pipeline computed.
        trace_raw = result.get("trace", {}) or {}
        trace = {
            "category": category,
            "review": review,
            "raw_novelty_text": default_run.get("output_novelty_text", ""),
            "idea_keywords": trace_raw.get("idea_keywords", []),
            "title_keywords": trace_raw.get("title_keywords", []),
            "idea_priority_facets": trace_raw.get("idea_priority_facets", ""),
            "snippet_papers": _df_to_records(trace_raw.get("snippet_papers")),
            "keyword_papers": _df_to_records(trace_raw.get("keyword_papers")),
            "embedding_ranked": _df_to_records(trace_raw.get("embedding_ranked")),
            "most_relevant_papers": _df_to_records(trace_raw.get("most_relevant_papers")),
            "evaluation_papers": default_run.get("evaluation_papers", []),
            # Scideator runs three separately-configured models plus an
            # embedder. Recorded explicitly so a run directory answers "which
            # model produced this row" without reading the source or the env.
            "models": {
                "novelty_check": llm_engine,
                "keyword_extraction": llm_engine,
                "rankgpt": rankgpt_model or llm_engine,
                "embedding": "allenai/specter2_base+specter2",
            },
            "config": {
                "model": llm_engine,
                "use_retrieval": use_retrieval,
                "novelty_check_prompt": novelty_check_prompt,
                "novelty_check_examples": novelty_check_examples,
                "novelty_check_topk": novelty_check_topk,
                "query_retrieval_method": query_retrieval_method,
                "rankgpt_model": rankgpt_model or llm_engine,
                "rankgpt_variant": rankgpt_variant,
            },
        }
        if ablation:
            trace["ablation_experiments"] = {
                exp_name: {
                    "category": exp_data.get("category", ""),
                    "review": exp_data.get("review", ""),
                    "evaluation_papers": exp_data.get("evaluation_papers", []),
                }
                for exp_name, exp_data in out_map.items()
            }

        return PointwiseResult(prediction=prediction, trace=trace)

