from __future__ import annotations

import re
from typing import List, Tuple, Iterable

from .transcriber import Segment


# Common filler tokens for English kitchen talk
_FILLER_PATTERNS = [
	r"\bso\b",
	r"\bum\b",
	r"\buh\b",
	r"\bokay\b",
	r"\bok\b",
	r"\blike\b",
	r"\byou know\b",
	r"\bkind of\b",
	r"\bsort of\b",
]
_F_RE = re.compile("|".join(_FILLER_PATTERNS), flags=re.IGNORECASE)
_REPEAT_RE = re.compile(r"\b(\w+)(\s+\1\b){1,}", flags=re.IGNORECASE)
_SPACE_RE = re.compile(r"\s{2,}")
_COMMA_RE = re.compile(r"(,)[,\s]{1,}")
_BRACKET_RE = re.compile(r"[\[\(][^\]\)]{0,30}[\]\)]")


def remove_filler_words(text: str) -> str:
	"""Lowercase safe removal of common fillers and collapse repeats."""
	t = _F_RE.sub("", text)
	t = _REPEAT_RE.sub(r"\1", t)
	# collapse repeated commas and remove stray spaces before punctuation
	t = _COMMA_RE.sub(r"\1 ", t)
	t = re.sub(r"\s+([,.;:!?])", r"\1", t)
	# remove short bracketed annotations like [music], (laughs)
	t = _BRACKET_RE.sub("", t)
	# remove leading/trailing punctuation leftovers
	t = re.sub(r"^[\s,.;:!?]+", "", t)
	t = re.sub(r"[\s,.;:!?]+$", "", t)
	t = _SPACE_RE.sub(" ", t)
	return t.strip()


def remove_filler_from_segments(segments: Iterable[Segment]) -> List[Segment]:
	out: List[Segment] = []
	for seg in segments:
		new_text = remove_filler_words(seg.text)
		out.append(Segment(index=seg.index, start=seg.start, end=seg.end, text=new_text))
	return out


_FILLER_TOKENS = {
	"so",
	"um",
	"uh",
	"okay",
	"ok",
	"like",
	"you",
	"know",
	"i",
	"mean",
	"that",
}


def evaluate_text_quality(text: str) -> Tuple[float, int]:
	"""Return (non_filler_ratio, total_words). Lower ratio suggests mostly filler.

	This is heuristic and only used to decide whether to attempt fallbacks.
	"""
	words = re.findall(r"[a-zA-Z']+", text.lower())
	if not words:
		return 0.0, 0
	non_filler = sum(1 for w in words if w not in _FILLER_TOKENS)
	return (non_filler / len(words)), len(words)


