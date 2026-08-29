"""Adapter between `run_benchmark.py`'s judge-backend seam and a `BaselineRunner`.

The seam is deliberately narrow: **a baseline returns a prediction, nothing else.**
Everything above it — the run loop, `scores.json`, metric accumulation, reports, the
cost report, `update_report` — stays in `run_benchmark.py`, exactly as it is for the
`standard` judge backend.

Deliberately *not* here:

- **Related-work marshalling.** AI-Scientist and Scideator both ignore the in-house
  retrieval cache and retrieve for themselves, so there is nothing to format.
- **A `prediction: -1` fallback.** A failed instance returns `None` and is omitted
  from `scores.json`, which is what lets `analysis/filtering.py` inject it as a wrong
  prediction. Writing a sentinel would put the instance in `present_pids` and defeat
  that safeguard.

Deliberately *is* here: **resume**. `scores.json` is only written once every
instance has finished, so a run killed partway leaves its per-instance traces on
disk with nothing downstream able to read them. `resume_dir` replays those
verdicts instead of re-paying for them.
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from ..comparison_logger import format_candidates_readable, write_pointwise_to_file
from ._shared.llm_adapters import (
    BaselineCallContext,
    reset_call_context,
    set_call_context,
)
from ._shared.s2_client import (
    RetrievalStats,
    reset_exclude_titles,
    reset_retrieval_stats,
    set_exclude_titles,
    set_retrieval_stats,
)
from .registry import available_baselines, get_runner_class

try:
    from logging_utils import setup_logger
    LOGGER = setup_logger(output_dir=os.getenv("OUTPUT_DIR", "."))
except Exception:  # pragma: no cover - logging must never break a run
    import logging
    LOGGER = logging.getLogger(__name__)


def extract_pointwise_exclude_titles(data: Dict[str, Any]) -> List[str]:
    """Titles a baseline's own retrieval must not return for this instance.

    Without this the test paper can retrieve *itself*, and every baseline that
    searches S2 would trivially call the idea unoriginal.
    """
    md = data.get("metadata") or {}
    titles: List[str] = []
    for key in ("title", "paper_title", "name"):
        value = md.get(key)
        if value:
            titles.append(str(value))
    explicit = md.get("exclude_titles") or md.get("aliases") or []
    if isinstance(explicit, list):
        titles.extend(str(t) for t in explicit if t)

    seen: set[str] = set()
    out: List[str] = []
    for title in titles:
        key = title.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(title)
    return out


class BaselineAdapter:
    """Wraps a `BaselineRunner` in the contract `run_benchmark.py` expects."""

    def __init__(
        self,
        name: str,
        *,
        baseline_kwargs: Optional[Dict[str, Any]] = None,
        cutoff_date: Optional[str] = None,
        no_topic: bool = False,
        resume_dir: Optional[str] = None,
    ):
        if name not in available_baselines():
            raise ValueError(
                f"Unknown baseline '{name}'. Available: {available_baselines()}"
            )
        self.name = name
        self.baseline_kwargs = dict(baseline_kwargs or {})
        self.cutoff_date = cutoff_date
        self.no_topic = no_topic
        self.resume_dir = _validate_resume_dir(resume_dir) if resume_dir else None
        self.runner = get_runner_class(name)(config=self.baseline_kwargs)
        self._summary: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Run-level artifacts
    # ------------------------------------------------------------------

    def _record_instance(
        self,
        instance_id: str,
        result: Any,
        retrieval: Dict[str, Any],
        output_path: str,
        llm_engine: str,
    ) -> None:
        """Append this instance to the run summary and rewrite it.

        Rewritten after every instance rather than once at the end: the adapter
        has no end-of-run hook, and a run that dies partway still leaves a
        usable summary. The file is small enough that the rewrite is free.
        """
        if not output_path:
            return
        trace = getattr(result, "trace", None) or {}
        self._summary.append({
            "instance_id": instance_id,
            "prediction": getattr(result, "prediction", None),
            "retrieval": retrieval,
            "models": trace.get("models", {}),
            # Baseline-specific signals worth watching across a run.
            "category": trace.get("category"),
            "exhausted_rounds": trace.get("exhausted_rounds"),
            "n_rounds": trace.get("n_rounds"),
            "papers_judged": len(trace.get("evaluation_papers") or []) or None,
        })
        try:
            _write_run_summary(self.name, llm_engine, self._summary, output_path)
        except Exception as exc:  # pragma: no cover — reporting must not kill a run
            LOGGER.warning(f"Could not write baseline run summary: {exc}")

    def _resume(self, instance_id: str, output_path: str) -> Optional[Any]:
        """Return a prior run's result for this instance, or None to run it now.

        The run directory's basename is mirrored under `resume_dir` because with
        `n > 1` each `run_pointwise_<i>` is an independent sample — resuming run 1
        from run 0's verdicts would collapse the runs into one.
        """
        if not self.resume_dir or not output_path:
            return None
        path = os.path.join(
            self.resume_dir, os.path.basename(output_path), "traces", f"{instance_id}.json",
        )
        try:
            with open(path) as fh:
                payload = json.load(fh)
        except FileNotFoundError:
            return None
        except Exception as exc:
            LOGGER.warning(f"Could not read resume trace {path}: {exc}")
            return None
        if payload.get("prediction") is None:
            # No verdict last time — re-run it. Resuming recovers cost, it does
            # not freeze failures in.
            return None
        return SimpleNamespace(
            prediction=payload["prediction"], trace=payload.get("trace") or {},
        )

    # ------------------------------------------------------------------
    # Construction from a run_benchmark config Namespace
    # ------------------------------------------------------------------

    @classmethod
    def from_args(cls, args: Any) -> "BaselineAdapter":
        """Build from the config Namespace, including the env vars vendored code reads.

        Keeps `run_benchmark.py`'s backend-selection branch to a couple of lines.
        """
        name = getattr(args, "baseline", None)
        if not name:
            raise ValueError("judge_backend 'baseline' requires a 'baseline' config field.")
        if getattr(args, "use_batch_api", False):
            raise ValueError(
                "judge_backend 'baseline' does not support use_batch_api. Baselines run "
                "sync only, because the iterative ones need each round's response before "
                "they can build the next request. Set use_batch_api: false."
            )

        cutoff_date = getattr(args, "cutoff_date", None)
        _mirror_s2_env(cutoff_date)

        baseline_kwargs = dict(getattr(args, "baseline_kwargs", {}) or {})
        resume_dir = baseline_kwargs.pop("resume_dir", None)
        if cutoff_date:
            baseline_kwargs.setdefault("cutoff_date", cutoff_date)

        adapter = cls(
            name,
            baseline_kwargs=baseline_kwargs,
            cutoff_date=cutoff_date,
            no_topic=getattr(args, "no_topic", False),
            resume_dir=resume_dir,
        )
        LOGGER.info(
            f"Judge backend: baseline '{name}' "
            f"(supports={sorted(adapter.runner.supports)}, cutoff_date={cutoff_date}"
            + (f", resume_dir={resume_dir}" if resume_dir else "") + ")"
        )
        return adapter

    # ------------------------------------------------------------------
    # Pointwise
    # ------------------------------------------------------------------

    async def pointwise(
        self,
        instance_id: Any,
        data: Dict[str, Any],
        *,
        llm_engine: str,
        semaphore: asyncio.Semaphore,
        effort: str = "medium",
        failure_log: Any = None,
        run_label: str = "",
        output_path: str = "",
    ) -> Optional[int]:
        """Return 0/1 for one instance, or None if the baseline could not decide.

        `None` means the instance is omitted from `scores.json` and later injected
        as a wrong prediction by the ablation layer.
        """
        if "pointwise" not in self.runner.supports:
            raise ValueError(f"Baseline '{self.name}' does not support pointwise.")

        resumed = self._resume(str(instance_id), output_path)
        if resumed is not None:
            # Re-record into *this* run directory so it ends up self-contained,
            # as if the instance had run here.
            retrieval = resumed.trace.get("retrieval_stats") or {}
            self._record_instance(
                str(instance_id), resumed, retrieval, output_path, llm_engine,
            )
            _persist_trace(resumed, str(instance_id), output_path, data.get("idea", ""))
            LOGGER.info(
                f"Instance {instance_id}: reused prediction {resumed.prediction} from "
                f"{self.resume_dir} (no LLM calls)."
            )
            return int(resumed.prediction)

        kwargs = dict(self.baseline_kwargs)
        kwargs.setdefault("exclude_titles", extract_pointwise_exclude_titles(data))

        ctx = BaselineCallContext(instance_id=str(instance_id), run_label=run_label)
        token = set_call_context(ctx)
        # Baselines whose vendored code has no route for `exclude_titles` — the
        # kwarg lands in their **_ and is dropped — pick it up from here instead,
        # so a test paper cannot retrieve itself. Passing it explicitly as well
        # keeps the baselines that do accept it unchanged.
        excl_token = set_exclude_titles(kwargs.get("exclude_titles"))
        # What S2 was asked for and what came back, so a verdict reached on a
        # thin set of papers is visible rather than indistinguishable from one
        # reached on a full set.
        stats = RetrievalStats()
        stats_token = set_retrieval_stats(stats)
        outcomes: List[Any] = []
        result = None
        try:
            # The runner ignores any `semaphore` kwarg (it lands in **_), so
            # concurrency has to be enforced here or `max_workers` would have no
            # effect and every instance would run at once.
            async with semaphore:
                result = await self.runner.run_pointwise(
                    str(instance_id),
                    data.get("idea", ""),
                    llm_engine=llm_engine,
                    topic=None if self.no_topic else data.get("context"),
                    effort=effort,
                    **kwargs,
                )
        except NotImplementedError:
            LOGGER.error(f"Baseline '{self.name}' does not implement pointwise.")
            return None
        except Exception as exc:
            LOGGER.warning(
                f"Baseline '{self.name}' raised on instance {instance_id}: "
                f"{type(exc).__name__}: {exc}"
            )
            return None
        finally:
            reset_call_context(token)
            reset_exclude_titles(excl_token)
            reset_retrieval_stats(stats_token)
            retrieval = stats.as_dict()
            if result is not None and isinstance(getattr(result, "trace", None), dict):
                result.trace["retrieval_stats"] = retrieval
            self._record_instance(
                str(instance_id), result, retrieval, output_path, llm_engine,
            )
            # Drained on the event loop: JudgeFailureLog is documented as
            # event-loop-only, but `chat` runs under asyncio.to_thread.
            outcomes = ctx.drain()
            _flush_outcomes(outcomes, ctx, failure_log)
            # Written before the checks below so a trace exists precisely when
            # the instance ends up with no prediction — that is when it is most
            # worth reading.
            _persist_trace(result, str(instance_id), output_path, data.get("idea", ""))

        # A baseline that never got a usable LLM response has not made a
        # judgement, whatever its default happens to be. AI-Scientist initialises
        # `novel = False` and returns it even when every round errored, which
        # would otherwise be recorded as a confident "not novel" — and score as
        # *correct* on every NEGATIVE instance. Treat it as a failure instead.
        if outcomes and not any(ok for ok, *_ in outcomes):
            LOGGER.warning(
                f"Baseline '{self.name}' made no successful LLM call for instance "
                f"{instance_id} ({len(outcomes)} failed); recording as no prediction."
            )
            return None

        if result is None or result.prediction is None:
            LOGGER.warning(
                f"Baseline '{self.name}' returned no prediction for instance {instance_id}."
            )
            return None
        return int(result.prediction)


def _validate_resume_dir(resume_dir: str) -> str:
    """Reject a `resume_dir` holding no traces: a resume that matches nothing
    re-pays for the whole run."""
    if not os.path.isdir(resume_dir):
        raise ValueError(f"resume_dir '{resume_dir}' is not a directory.")
    if not glob.glob(os.path.join(resume_dir, "run_*", "traces", "*.json")):
        raise ValueError(
            f"resume_dir '{resume_dir}' contains no run_*/traces/*.json. Point it at the "
            "timestamped artifacts directory of the interrupted run — the parent of "
            "run_pointwise_0/ — not at the run directory itself."
        )
    return resume_dir


def _persist_trace(result: Any, instance_id: str, output_path: str, idea_text: str) -> None:
    """Write the baseline's reasoning to disk, in two forms.

    Neither may break a run, hence the broad excepts — a missing trace is an
    annoyance, a crashed 300-instance run is not.

    1. One record per round in `<output_path>/pointwise/<instance_id>.txt`, via the
       same writer the in-house judge uses. `md_report_writer` already reads that
       file, so baseline rounds show up in the debug report with no report-side
       changes.
    2. The whole trace verbatim in `<output_path>/traces/<instance_id>.json`. The
       text form above flattens things; this keeps `msg_history` (the prompt as
       actually sent) and the full paper records.
    """
    trace = getattr(result, "trace", None)
    if not output_path or not trace:
        return

    try:
        for rnd in trace.get("rounds") or []:
            choice: Dict[str, Any] = {
                "round": rnd.get("round"),
                "query": rnd.get("query"),
                "decided": rnd.get("decided"),
                "papers_found": len(rnd.get("papers") or []),
            }
            if rnd.get("error"):
                choice["error"] = rnd["error"]
            write_pointwise_to_file(
                output_path,
                instance_id,
                idea_text,
                format_candidates_readable(rnd.get("papers") or []),
                rnd.get("raw_response") or "",
                choice,
                rnd.get("round", 0),
            )
    except Exception as exc:
        LOGGER.warning(f"Could not write pointwise log for instance {instance_id}: {exc}")

    try:
        traces_dir = os.path.join(output_path, "traces")
        os.makedirs(traces_dir, exist_ok=True)
        payload = {
            "instance_id": instance_id,
            "prediction": getattr(result, "prediction", None),
            "trace": trace,
        }
        with open(os.path.join(traces_dir, f"{instance_id}.json"), "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
    except Exception as exc:
        LOGGER.warning(f"Could not write trace for instance {instance_id}: {exc}")


def _flush_outcomes(outcomes: List[Any], ctx: BaselineCallContext, failure_log: Any) -> None:
    if failure_log is None:
        return
    from novelty_eval.judge import _JudgeFailure

    for ok, prompt, code, err_type, err_msg in outcomes:
        if ok:
            failure_log.record_success()
            continue
        failure_log.record_failure(_JudgeFailure(
            problem_name=ctx.instance_id,
            idea_i=ctx.instance_id,
            idea_j="",
            kind="llm_empty",
            prompt=prompt,
            raw_response="",
            error_code=code,
            error_type=err_type,
            error_msg=err_msg,
            run_label=ctx.run_label,
        ))


def _write_run_summary(
    baseline: str, llm_engine: str, rows: List[Dict[str, Any]], output_path: str,
) -> None:
    """Write baseline_summary.{json,md} — the post-run view of a baseline run.

    Answers, without opening 313 traces: which model ran each stage, how much
    retrieval each instance actually got, and which instances produced no
    prediction.
    """
    os.makedirs(output_path, exist_ok=True)

    def _sum(key: str) -> int:
        return sum((r["retrieval"] or {}).get(key, 0) for r in rows)

    n = len(rows)
    no_pred = [r for r in rows if r["prediction"] is None]
    fracs = [(r["retrieval"] or {}).get("retained_fraction", 1.0) for r in rows]
    thin = [r for r in rows if (r["retrieval"] or {}).get("retained_fraction", 1.0) < 0.9]
    models: Dict[str, Any] = {}
    for r in rows:
        models.update(r.get("models") or {})

    aggregate = {
        "baseline": baseline,
        "llm_engine": llm_engine,
        "instances": n,
        "no_prediction": len(no_pred),
        "no_prediction_ids": [r["instance_id"] for r in no_pred],
        "models": models,
        "retrieval_totals": {
            k: _sum(k) for k in (
                "search_calls", "papers_returned", "usable_papers",
                "excluded_by_leakage", "entries_without_corpus_id",
                "ids_requested", "ids_unresolved", "chunks_failed",
            )
        },
        "retained_fraction_min": round(min(fracs), 4) if fracs else None,
        "retained_fraction_mean": round(sum(fracs) / len(fracs), 4) if fracs else None,
        "instances_below_90pct_retained": [r["instance_id"] for r in thin],
    }

    with open(os.path.join(output_path, "baseline_summary.json"), "w") as fh:
        json.dump({"aggregate": aggregate, "instances": rows}, fh, indent=2, default=str)

    t = aggregate["retrieval_totals"]
    md = [
        f"# Baseline run summary — `{baseline}`", "",
        f"**Engine**: `{llm_engine}` · **Instances**: {n} · "
        f"**No prediction**: {len(no_pred)}", "",
        "## Models used, by stage", "",
        "| stage | model |", "| --- | --- |",
    ]
    md += [f"| {k} | `{v}` |" for k, v in sorted(models.items())] or ["| _(none recorded)_ | |"]
    md += [
        "", "## Retrieval", "",
        "| metric | value |", "| --- | --- |",
        f"| S2 calls | {t['search_calls']} |",
        f"| papers returned by S2 | {t['papers_returned']} |",
        f"| ...of which usable (carry a `corpusId`) | {t['usable_papers']} |",
        f"| ...of which unusable (no `corpusId`) | {t['entries_without_corpus_id']} |",
        f"| ids requested / never resolved | {t['ids_requested']} / {t['ids_unresolved']} |",
        f"| failed batch chunks | {t['chunks_failed']} |",
        f"| dropped by the leakage filter | {t['excluded_by_leakage']} |",
        f"| retained fraction (mean / min) | "
        f"{aggregate['retained_fraction_mean']} / {aggregate['retained_fraction_min']} |",
        "",
        "**usable** — returned by S2 with a top-level `corpusId`. Entries without one are",
        "discarded by the retrieval guards before ranking, so they never reach the judge.",
        "",
        "**retained fraction** — `usable / (returned + never resolved)`. Counts **unintended**",
        "loss only: papers removed by the leakage filter are that filter working as designed,",
        "and narrowing to the cosine top-k and then the judge's top-k is the algorithm itself.",
        "Neither is counted here, so 1.0 means nothing was lost that should not have been.",
        "",
    ]
    if thin:
        md += [
            f"> {len(thin)} instance(s) retained under 90% of what was requested — "
            "a verdict from these rested on less evidence than intended: "
            + ", ".join(r["instance_id"] for r in thin[:20])
            + (" ..." if len(thin) > 20 else ""), "",
        ]
    if no_pred:
        md += [
            f"> {len(no_pred)} instance(s) produced no prediction and are omitted from "
            "scores.json, so the ablation layer injects them as wrong: "
            + ", ".join(r["instance_id"] for r in no_pred[:20])
            + (" ..." if len(no_pred) > 20 else ""), "",
        ]
    with open(os.path.join(output_path, "baseline_summary.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")


def _mirror_s2_env(cutoff_date: Optional[str]) -> None:
    """Expose the S2 key and cutoff the way vendored baseline code expects.

    The vendored searchers read env vars rather than `secrets.toml`; without this
    they hit S2 unauthenticated and the anonymous rate limit produces a cascade of
    429/500s even with the shared limiter in place.
    """
    try:
        from utils import SECRETS
    except Exception:
        SECRETS = None

    key = SECRETS.get("semantic_scholar_key") if SECRETS else None
    if key:
        os.environ.setdefault("S2_API_KEY", key)
        # Upstream env var name; misspelling is theirs, see _typos.toml.
        os.environ.setdefault("SEMENTIC_SEARCH_API_KEY", key)
    else:
        LOGGER.warning(
            "No semantic_scholar_key in SECRETS; vendored baselines will hit S2 "
            "unauthenticated."
        )

    if cutoff_date:
        os.environ["S2_CUTOFF_DATE"] = str(cutoff_date)
    else:
        os.environ.pop("S2_CUTOFF_DATE", None)
