"""
JEV Lab 共通スキーマ

Jev 級（型付き判断）モデルの入出力を、モデル実装に依存しない形で定義する。

  state  +  [ Choice(instr, options) | Score(instr, levels) | Noul(instr) ]
     -> [ {choice, probabilities, confidence} | {score, probabilities, confidence} | {noul, p_yes} ]

すべての question は「選択肢集合上の 1 つの softmax」として扱う:
  Choice : options 上の分布            -> argmax
  Score  : levels（順序付き）上の分布   -> 期待値 E[level]
  Noul   : {"yes","no"} 上の分布        -> p(yes)
"""
from __future__ import annotations

from typing import Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------- questions
class ChoiceQ(BaseModel):
    type: Literal["choice"] = "choice"
    id: str
    instructions: str
    options: list[str] = Field(min_length=2, max_length=255)
    # 選択肢に補足説明を付けたい場合（GLiNER2 の "labels with a description" 相当）
    descriptions: Optional[dict[str, str]] = None


class ScoreQ(BaseModel):
    type: Literal["score"] = "score"
    id: str
    instructions: str
    levels: list[str] = Field(min_length=2, max_length=10)  # 低→高の順
    # 数値として読む場合の値。省略時は 0..K-1
    values: Optional[list[float]] = None

    @model_validator(mode="after")
    def _check(self):
        if self.values is not None and len(self.values) != len(self.levels):
            raise ValueError("values と levels の長さが一致しません")
        return self


class NoulQ(BaseModel):
    type: Literal["noul"] = "noul"
    id: str
    instructions: str  # yes/no で答えられる問い


Question = Union[ChoiceQ, ScoreQ, NoulQ]

NOUL_OPTIONS = ["yes", "no"]


def options_of(q: Question) -> list[str]:
    """あらゆる question を「選択肢リスト」に正規化する。"""
    if q.type == "choice":
        return q.options
    if q.type == "score":
        return q.levels
    return NOUL_OPTIONS


# ---------------------------------------------------------------- request
class DecideRequest(BaseModel):
    state: str = Field(description="判断対象の文脈（program state）")
    questions: list[Question] = Field(min_length=1)
    # RAG 等で注入した根拠文脈。state に連結してモデルへ渡す
    context: Optional[list[str]] = None
    model: Optional[str] = None  # models.yaml のキー。省略時は default


# ---------------------------------------------------------------- results
class Distribution(BaseModel):
    labels: list[str]
    probabilities: list[float]

    @model_validator(mode="after")
    def _check(self):
        if len(self.labels) != len(self.probabilities):
            raise ValueError("labels と probabilities の長さが一致しません")
        s = sum(self.probabilities)
        if s <= 0:
            raise ValueError("確率の総和が 0 以下です")
        # JevBench の方針: 総和が 1 から 2% 以内なら正規化、それ以外は invalid
        if abs(s - 1.0) > 0.02:
            raise ValueError(f"確率の総和が 1 から 2% 以上ずれています: {s:.4f}")
        self.probabilities = [p / s for p in self.probabilities]
        return self

    @property
    def argmax(self) -> int:
        return max(range(len(self.probabilities)), key=lambda i: self.probabilities[i])

    @property
    def confidence(self) -> float:
        return max(self.probabilities)


class ChoiceA(BaseModel):
    type: Literal["choice"] = "choice"
    id: str
    choice: str
    confidence: float
    distribution: Distribution


class ScoreA(BaseModel):
    type: Literal["score"] = "score"
    id: str
    score: float           # 期待値（段階の間に落ちることがある。例 1.035）
    level: str             # argmax の段階名
    confidence: float
    distribution: Distribution


class NoulA(BaseModel):
    type: Literal["noul"] = "noul"
    id: str
    noul: bool
    p_yes: float
    distribution: Distribution


Answer = Union[ChoiceA, ScoreA, NoulA]


class DecideResponse(BaseModel):
    model: str
    answers: list[Answer]
    latency_ms: float
    # "native": モデル自身の分布 / "verbalized": LLM が書いた分布（JevBench の区別を踏襲）
    probability_kind: Literal["native", "verbalized"]
    notes: Optional[str] = None


# ---------------------------------------------------------------- helpers
def answer_from_distribution(q: Question, probs: list[float]) -> Answer:
    """選択肢上の確率ベクトルから、question 型に応じた Answer を組み立てる。"""
    labels = options_of(q)
    dist = Distribution(labels=labels, probabilities=probs)
    if q.type == "choice":
        i = dist.argmax
        return ChoiceA(id=q.id, choice=labels[i], confidence=dist.confidence, distribution=dist)
    if q.type == "score":
        values = q.values if q.values is not None else [float(k) for k in range(len(labels))]
        expected = sum(p * v for p, v in zip(dist.probabilities, values))
        i = dist.argmax
        return ScoreA(id=q.id, score=expected, level=labels[i], confidence=dist.confidence, distribution=dist)
    p_yes = dist.probabilities[0]
    return NoulA(id=q.id, noul=p_yes >= 0.5, p_yes=p_yes, distribution=dist)


def build_state(state: str, context: Optional[list[str]]) -> str:
    """RAG 文脈を state に連結する。注入位置は state の前（根拠→状況の順）。"""
    if not context:
        return state
    ctx = "\n".join(f"- {c}" for c in context)
    return f"[根拠]\n{ctx}\n[状況]\n{state}"
