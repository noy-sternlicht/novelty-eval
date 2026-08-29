"""Baseline LLM calls must reach the cost tracker tagged with their pipeline
stage, otherwise the cost report's by-stage table is all `unknown`."""
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import cost_tracker  # noqa: E402
from novelty_eval.baselines._shared import llm_adapters  # noqa: E402


def _stage_seen_by(client, call, monkeypatch):
    """The stage a `prompt_openai_client` call would be recorded under."""
    seen = {}

    def fake(prompt, **kwargs):
        seen["stage"] = cost_tracker._current_stage.get()
        return "ok"

    monkeypatch.setattr(llm_adapters, "prompt_openai_client", fake)
    call(client)
    return seen["stage"]


def test_chat_tags_its_stage(monkeypatch):
    client = llm_adapters.SharedLLMClient(model="gpt-5.1", stage="scideator_rankgpt")
    stage = _stage_seen_by(client, lambda c: c.chat("idea"), monkeypatch)
    assert stage == "scideator_rankgpt"


def test_a_chat_tags_across_the_worker_thread(monkeypatch):
    """a_chat hops threads via asyncio.to_thread; the ContextVar must survive."""
    client = llm_adapters.SharedLLMClient(model="gpt-5.1", stage="scideator_verdict")
    stage = _stage_seen_by(
        client, lambda c: asyncio.run(c.a_chat("idea")), monkeypatch,
    )
    assert stage == "scideator_verdict"


def test_stage_does_not_leak_after_the_call(monkeypatch):
    client = llm_adapters.SharedLLMClient(model="gpt-5.1", stage="scideator_keywords")
    _stage_seen_by(client, lambda c: c.chat("idea"), monkeypatch)
    assert cost_tracker._current_stage.get() == "unknown"


def test_every_baseline_call_site_is_tagged():
    """A new client built without `stage` would silently land in the default."""
    baselines = Path(__file__).resolve().parents[1] / "src/novelty_eval/baselines"
    untagged = [
        f"{path.relative_to(baselines)}:{i}"
        for path in baselines.rglob("*.py")
        for i, line in enumerate(path.read_text().splitlines(), 1)
        if "SharedLLMClient(" in line and "stage=" not in line and "class " not in line
    ]
    assert not untagged, f"SharedLLMClient built without a cost stage: {untagged}"
