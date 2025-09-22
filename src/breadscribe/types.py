from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Segment:
	index: int
	start: float
	end: float
	text: str


@dataclass
class TranscriptResult:
	detected_language: Optional[str]
	segments: List[Segment]
	text: str
	rtf: Optional[float]
	used_stream_index: Optional[int] = None


