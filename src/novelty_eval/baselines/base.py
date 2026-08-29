from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional

Mode = Literal["pointwise", "pairwise", "ranking"]


@dataclass
class PointwiseResult:
    prediction: Optional[int]         # 0 = not novel, 1 = novel, None = no verdict
    label: Optional[str] = None       # POSITIVE | NEGATIVE if known
    raw_score: Optional[float] = None # underlying continuous score, if any
    trace: dict[str, Any] = field(default_factory=dict)

    def to_scores_entry(self) -> dict:
        out = {"prediction": self.prediction}
        if self.label is not None:
            out["label"] = self.label
        if self.raw_score is not None:
            out["raw_score"] = self.raw_score
        if self.trace:
            out["trace"] = self.trace
        return out


@dataclass
class PairwiseResult:
    """Result of one pair comparison.

    `winner` follows the existing convention used by main.judge_idea:
        0 → idea0 wins, 1 → idea1 wins, 2 → tie, None → error.
    """
    winner: Optional[int]
    scores: dict[str, int] = field(default_factory=dict)   # {dim: 0|1|2}
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass
class RankingResult:
    """Result of ranking all ideas in one problem."""
    ideas: list[Any]
    elo_scores: list[float]
    comparisons: list[dict] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)


class BaselineRunner(ABC):
    """Subclasses set `name`, `paper`, `supports`, and override the modes
    listed in `supports`. Modes outside `supports` raise NotImplementedError
    by default.
    """
    name: str = ""
    paper: str = ""
    supports: set[Mode] = set()

    def __init__(self, config: dict | None = None):
        self.config: dict = config or {}

    async def run_pointwise(
        self,
        instance_id: str,
        idea: str,
        *,
        llm_engine: str,
        retrieval_cache_entry: Optional[dict] = None,
        topic: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[PointwiseResult]:
        raise NotImplementedError(f"{self.name} does not support pointwise")

    async def run_pairwise(
        self,
        problem_id: str,
        idea0: dict,
        idea1: dict,
        *,
        llm_engine: str,
        retrieval_cache_entry: Optional[dict] = None,
        topic: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[PairwiseResult]:
        raise NotImplementedError(f"{self.name} does not support pairwise")

    async def run_ranking(
        self,
        problem_id: str,
        ideas: dict,
        *,
        llm_engine: str,
        retrieval_cache: Optional[dict] = None,
        topic: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[RankingResult]:
        raise NotImplementedError(f"{self.name} does not support ranking")

    def supports_mode(self, mode: Mode) -> bool:
        return mode in self.supports
