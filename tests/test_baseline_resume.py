"""Resume path of BaselineAdapter — a resume that matches nothing has no symptom
other than the API bill, so every branch of it is pinned here."""
import asyncio
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

# No `src.` prefix, unlike other tests: registry.py imports
# `novelty_eval...`, and the two spellings give two module
# objects, so `issubclass(cls, BaselineRunner)` would fail.
from novelty_eval.baselines.adapter import (  # noqa: E402
    BaselineAdapter,
    _validate_resume_dir,
)


def _write_trace(run_dir: Path, instance_id: str, prediction):
    traces = run_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    (traces / f"{instance_id}.json").write_text(json.dumps({
        "instance_id": instance_id,
        "prediction": prediction,
        "trace": {
            "models": {"novelty": "gpt-5.1"},
            "category": "novel",
            "n_rounds": 3,
            "retrieval_stats": {"search_calls": 3, "retained_fraction": 1.0},
            "rounds": [{"round": 0, "query": "q", "papers": [], "raw_response": "r"}],
        },
    }))


@pytest.fixture
def runs(tmp_path):
    """An interrupted run (A decided, B did not) and a fresh run directory."""
    old_run = tmp_path / "old" / "run_pointwise_0"
    new_run = tmp_path / "new" / "run_pointwise_0"
    new_run.mkdir(parents=True)
    _write_trace(old_run, "A", 1)
    _write_trace(old_run, "B", None)
    return tmp_path / "old", new_run


def _adapter(resume_dir=None):
    kwargs = {"max_num_iterations": 10}
    if resume_dir is not None:
        kwargs["resume_dir"] = str(resume_dir)
    return BaselineAdapter.from_args(Namespace(
        baseline="ai_scientist",
        use_batch_api=False,
        cutoff_date="2025-03-01",
        baseline_kwargs=kwargs,
        no_topic=False,
    ))


def test_resume_dir_does_not_reach_the_runner(runs):
    """Leaking it would hand vendored code a stray kwarg."""
    old_dir, _ = runs
    adapter = _adapter(old_dir)
    assert adapter.resume_dir == str(old_dir)
    assert "resume_dir" not in adapter.baseline_kwargs


def test_decided_instance_is_replayed_without_llm_calls(runs):
    old_dir, new_run = runs
    adapter = _adapter(old_dir)

    prediction = asyncio.run(adapter.pointwise(
        "A",
        {"idea": "some idea", "metadata": {"title": "T"}},
        llm_engine="gpt-5.1",
        semaphore=asyncio.Semaphore(1),
        output_path=str(new_run),
    ))

    assert prediction == 1
    # The new run directory must stand on its own.
    assert (new_run / "traces" / "A.json").exists()
    assert (new_run / "pointwise").is_dir()
    summary = json.loads((new_run / "baseline_summary.json").read_text())
    row = summary["instances"][0]
    assert row["prediction"] == 1
    assert row["retrieval"]["search_calls"] == 3
    assert summary["aggregate"]["models"] == {"novelty": "gpt-5.1"}


def test_instances_without_a_verdict_are_rerun(runs):
    """Resuming recovers cost; it must not freeze failures in."""
    old_dir, new_run = runs
    assert _adapter(old_dir)._resume("B", str(new_run)) is None


def test_missing_trace_is_rerun(runs):
    old_dir, new_run = runs
    assert _adapter(old_dir)._resume("never-ran", str(new_run)) is None


def test_runs_only_resume_from_their_own_counterpart(runs):
    """With n > 1 each run is an independent sample."""
    old_dir, new_run = runs
    other_run = new_run.parent / "run_pointwise_1"
    assert _adapter(old_dir)._resume("A", str(other_run)) is None


def test_no_resume_dir_configured(runs):
    _, new_run = runs
    assert _adapter()._resume("A", str(new_run)) is None


def test_unusable_resume_dir_is_an_error(tmp_path, runs):
    old_dir, _ = runs
    with pytest.raises(ValueError):
        _validate_resume_dir(str(tmp_path / "does-not-exist"))
    # The run directory rather than its parent.
    with pytest.raises(ValueError, match="run_pointwise_0"):
        _validate_resume_dir(str(old_dir / "run_pointwise_0"))
