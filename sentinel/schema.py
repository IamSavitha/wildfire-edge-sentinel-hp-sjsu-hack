"""Shared data types for the sentinel pipeline."""
from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

from pydantic import BaseModel, Field


class Severity(IntEnum):
    IGNORE = 0
    LOG = 1
    MONITOR = 2
    ALERT = 3


@dataclass(frozen=True)
class Detection:
    cls: str
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


SourceType = Literal[
    "wildland", "structure", "vehicle", "controlled_burn", "campfire",
    "bbq_chimney", "industrial_stack", "fog_dust_cloud", "unknown",
]


class ContextResult(BaseModel):
    source_type: SourceType
    smoke_color: Literal["white", "grey", "black", "none"]
    attended: Literal["yes", "no", "unclear"]
    near_structures: bool
    near_road: bool
    size_estimate: Literal["small", "medium", "large"]
    description: str = Field(max_length=200)


CONTEXT_JSON_SCHEMA = ContextResult.model_json_schema()
