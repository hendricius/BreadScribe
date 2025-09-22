from __future__ import annotations

from typing import List
from .types import Segment


def _format_timestamp_srt(seconds: float) -> str:
	if seconds < 0:
		seconds = 0.0
	ms = int(round(seconds * 1000))
	hours = ms // 3_600_000
	ms %= 3_600_000
	minutes = ms // 60_000
	ms %= 60_000
	secs = ms // 1000
	ms %= 1000
	return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
	if seconds < 0:
		seconds = 0.0
	ms = int(round(seconds * 1000))
	hours = ms // 3_600_000
	ms %= 3_600_000
	minutes = ms // 60_000
	ms %= 60_000
	secs = ms // 1000
	ms %= 1000
	return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def segments_to_srt(segments: List[Segment]) -> str:
	lines: list[str] = []
	for i, seg in enumerate(segments, start=1):
		start = _format_timestamp_srt(seg.start)
		end = _format_timestamp_srt(seg.end)
		lines.append(str(i))
		lines.append(f"{start} --> {end}")
		lines.append(seg.text.strip())
		lines.append("")
	return "\n".join(lines).strip() + "\n"


def segments_to_vtt(segments: List[Segment]) -> str:
	lines: list[str] = ["WEBVTT", ""]
	for seg in segments:
		start = _format_timestamp_vtt(seg.start)
		end = _format_timestamp_vtt(seg.end)
		lines.append(f"{start} --> {end}")
		lines.append(seg.text.strip())
		lines.append("")
	return "\n".join(lines).strip() + "\n"



