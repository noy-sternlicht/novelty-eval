"""Thin LLM-client adapter presenting the `chat` / `a_chat` interface the
vendored baselines expect, while routing through the shared
`prompt_openai_client` transport in `src/utils.py`.

This is what lets us keep vendored baseline code (`check_novelty.py` etc.)
byte-identical while still getting unified cost tracking and key handling.
"""
from __future__ import annotations

import asyncio
import contextvars
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, Optional, Tuple

from utils import prompt_openai_client

try:
    from cost_tracker import cost_stage
except Exception:  # pragma: no cover — cost tagging must never break a run
    from contextlib import nullcontext as cost_stage


@dataclass
class BaselineCallContext:
    """Per-instance context for LLM calls made deep inside vendored baseline code.

    Set by the adapter around a single instance's run. Carried through
    `asyncio.to_thread` automatically, since that propagates the current
    contextvars Context, and scoped per task because `asyncio.gather` copies the
    context at task creation.

    Outcomes are *buffered* rather than written straight to `JudgeFailureLog`:
    `chat` runs on a worker thread, and `JudgeFailureLog` documents itself as
    event-loop-only (`self.total += 1` is a non-atomic read-modify-write). The
    adapter drains this back on the loop once the instance finishes.
    """

    instance_id: str = ""
    run_label: str = ""
    # (ok, prompt, error_code, error_type, error_msg)
    outcomes: List[Tuple[bool, str, str, str, str]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, ok: bool, prompt: str, err: Mapping[str, Any]) -> None:
        code = err.get("code")
        with self._lock:
            self.outcomes.append((
                ok,
                prompt,
                "" if code is None else str(code),
                err.get("type", ""),
                err.get("msg", ""),
            ))

    def drain(self) -> List[Tuple[bool, str, str, str, str]]:
        with self._lock:
            out, self.outcomes = self.outcomes, []
        return out


_CALL_CONTEXT: contextvars.ContextVar[Optional[BaselineCallContext]] = contextvars.ContextVar(
    "baseline_call_context", default=None,
)


def set_call_context(ctx: Optional[BaselineCallContext]):
    """Install the per-instance context; returns a token for `reset_call_context`."""
    return _CALL_CONTEXT.set(ctx)


def reset_call_context(token) -> None:
    _CALL_CONTEXT.reset(token)


def _readable(messages: Iterable[Mapping[str, Any]]) -> str:
    """Render a message array as text, for the failure log only.

    The model is sent the real array; this is purely so a recorded failure shows
    what was asked rather than a repr.
    """
    parts: List[str] = []
    for m in messages:
        role = (m.get("role") or "user").upper()
        content = m.get("content") or ""
        if isinstance(content, list):
            # Anthropic-style content blocks → join text parts
            content = "\n".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in content
            )
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


class SharedLLMClient:
    """Presents `chat` / `a_chat`, the interface AI Scientist and Scideator use."""

    def __init__(self, model: str = "gpt-5.1", effort: Optional[str] = None, stage: str = "baseline"):
        self.model = model
        # Vendored code that constructs this client itself cannot pass `effort`
        # without diverging from upstream, so it falls back to the environment.
        # Default is None, not "medium": a truthy default would shadow the env.
        self.effort = effort or os.getenv("BASELINE_REASONING_EFFORT") or "medium"
        self.stage = stage

    # ── AI Scientist / Scideator interface ────────────────────────────────
    def chat(
        self,
        messages: Any,
        *,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        return_history: bool = False,
        msg_history: Optional[List[Mapping[str, Any]]] = None,
        temperature: Optional[float] = None,  # accepted, ignored — main.py also ignores
        **_: Any,
    ):
        msg_history = list(msg_history or [])
        if isinstance(messages, str):
            new_msg = msg_history + [{"role": "user", "content": messages}]
        else:
            new_msg = msg_history + list(messages)

        if system_message:
            payload = [{"role": "system", "content": system_message}] + new_msg
        else:
            payload = new_msg
        # Mirrors the judge, which passes an `error_out` list and feeds the
        # captured code/type/msg into JudgeFailureLog. Without it, baseline LLM
        # failures are invisible and `response or ""` hands the caller an empty
        # string rather than a failure signal.
        error_out: list = []

        with cost_stage(self.stage):
            response = prompt_openai_client(
                payload,
                engine=model or self.model,
                max_completion_tokens=None,
                reasoning={"effort": self.effort},
                error_out=error_out,
                use_prompt_caching=True,
            )

        ctx = _CALL_CONTEXT.get()
        if ctx is not None:
            ctx.record(bool(response), _readable(payload), error_out[0] if error_out else {})

        if return_history:
            updated_history = new_msg + [{"role": "assistant", "content": response or ""}]
            return response or "", updated_history
        return response or ""

    async def a_chat(self, *args, **kwargs):
        return await asyncio.to_thread(self.chat, *args, **kwargs)
