"""Evaluation dataset models."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class EvaluationQuery(BaseModel):
    query_id: str
    query_text: str
    relevance_judgments: dict[str, int] = Field(default_factory=dict)

    @field_validator("query_id", "query_text")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("value must be a non-empty string")
        return v

    @field_validator("relevance_judgments")
    @classmethod
    def valid_grades(cls, v: dict[str, int]) -> dict[str, int]:
        for cid, grade in v.items():
            if not isinstance(cid, str) or not cid:
                raise ValueError("chunk id must be a non-empty string")
            if not isinstance(grade, int) or grade < 0:
                raise ValueError("relevance grade must be a non-negative integer")
        return v


class EvaluationDataset(BaseModel):
    queries: list[EvaluationQuery] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [q.query_id for q in self.queries]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate query IDs")
        return self


def example_dataset() -> EvaluationDataset:
    """Illustrative example only; relevance judgments are not experimentally validated ground truth."""
    return EvaluationDataset(
        queries=[
            EvaluationQuery(
                query_id="q1",
                query_text="How do I reset my password?",
                relevance_judgments={"C1": 3, "C3": 2, "C5": 1},
            ),
            EvaluationQuery(
                query_id="q2",
                query_text="recover account access",
                relevance_judgments={"C2": 3, "C4": 1},
            ),
        ]
    )
