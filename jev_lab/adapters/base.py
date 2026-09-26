"""
アダプタ共通インターフェース。

全アダプタは decide(state, questions) -> DecideResponse を実装する。
1 state に N question を載せ、可能なら 1 forward で答える（Jev の形）。
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from ..schema import (
    Answer, DecideResponse, Question, answer_from_distribution, build_state,
)


class BaseAdapter(ABC):
    name: str
    probability_kind: str = "native"

    def __init__(self, name: str, cfg: dict[str, Any]):
        self.name = name
        self.cfg = cfg
        self._loaded = False

    # -- lifecycle -------------------------------------------------------
    def load(self) -> None:
        """重みのロード。初回 decide 時に遅延実行される。"""
        if not self._loaded:
            self._load()
            self._loaded = True

    @abstractmethod
    def _load(self) -> None: ...

    # -- core --------------------------------------------------------------
    @abstractmethod
    def distributions(self, state: str, questions: list[Question]) -> list[list[float]]:
        """question ごとの選択肢上の確率ベクトルを返す（順序は options_of と同じ）。"""

    def decide(self, state: str, questions: list[Question], context=None) -> DecideResponse:
        self.load()
        full_state = build_state(state, context)
        t0 = time.perf_counter()
        probs = self.distributions(full_state, questions)
        dt = (time.perf_counter() - t0) * 1000
        answers: list[Answer] = [answer_from_distribution(q, p) for q, p in zip(questions, probs)]
        return DecideResponse(
            model=self.name, answers=answers, latency_ms=dt,
            probability_kind=self.probability_kind,
        )

    # -- info --------------------------------------------------------------
    def info(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.probability_kind, **self.cfg}
