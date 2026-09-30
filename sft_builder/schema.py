"""
Data schemas for SFT samples.

Output format:
{
    "image": "path/url/handle",
    "instruction": "text",
    "response": {
        "score": 0.0,
        "text_reason": "...",
        "visual_reason": {"type": "mask|heatmap|bbox|polygon", "data": ...}
    },
    "meta": {"source": "...", "id": "...", "orig": {...}}
}
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional
import numpy as np


VisualReasonType = Literal["mask", "heatmap", "bbox", "polygon", "svg"]


@dataclass
class VisualReason:
    """Visual grounding for evaluation."""
    type: VisualReasonType
    data: Any  # mask: np.ndarray or path, heatmap: np.ndarray or path, bbox: List[x, y, w, h], polygon: List[List[x, y]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = self.data
        if isinstance(data, np.ndarray):
            data = data.tolist()
        return {"type": self.type, "data": data}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VisualReason":
        return cls(type=d["type"], data=d["data"])


@dataclass
class Response:
    """Evaluation response with text and visual grounding."""
    score: float
    text_reason: str
    visual_reason: VisualReason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "text_reason": self.text_reason,
            "visual_reason": self.visual_reason.to_dict()
        }


@dataclass
class Meta:
    """Metadata for traceability."""
    source: str
    id: str
    orig: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "id": self.id, "orig": self.orig}


@dataclass
class SFTSample:
    """Complete SFT sample with visual grounding."""
    image: str  # path, URL, or handle
    instruction: str
    response: Response
    meta: Meta

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image": self.image,
            "instruction": self.instruction,
            "response": self.response.to_dict(),
            "meta": self.meta.to_dict()
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SFTSample":
        return cls(
            image=d["image"],
            instruction=d["instruction"],
            response=Response(
                score=d["response"]["score"],
                text_reason=d["response"]["text_reason"],
                visual_reason=VisualReason.from_dict(d["response"]["visual_reason"])
            ),
            meta=Meta(
                source=d["meta"]["source"],
                id=d["meta"]["id"],
                orig=d["meta"].get("orig", {})
            )
        )


@dataclass
class RawSample:
    """
    Intermediate representation before validation.

    Fields may be None/invalid before validation passes.
    """
    image: Optional[str]
    instruction: Optional[str]
    score: Optional[float]
    text_reason: Optional[str]
    visual_reason_type: Optional[VisualReasonType]
    visual_reason_data: Any
    source: str
    id: str
    orig: Dict[str, Any] = field(default_factory=dict)

    # Image dimensions for validation
    image_width: Optional[int] = None
    image_height: Optional[int] = None
