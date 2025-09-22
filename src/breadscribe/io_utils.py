from __future__ import annotations

from pathlib import Path
from typing import Iterable
import csv


ALLOWED_EXTS: set[str] = {
	".mp4", ".mov", ".mkv", ".m4v", ".avi", ".flv",
	".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac",
}


def discover_inputs(path: Path, recursive: bool, allowed_exts: set[str], include_hidden: bool = False) -> list[Path]:
	results: list[Path] = []
	if path.is_file():
		if path.suffix.lower() in allowed_exts:
			results.append(path)
		return results
	if not path.is_dir():
		return results
	glob = "**/*" if recursive else "*"
	for p in path.glob(glob):
		if p.is_file() and p.suffix.lower() in allowed_exts:
			if not include_hidden:
				try:
					rel = p.relative_to(path)
					if any(part.startswith(".") for part in rel.parts):
						continue
				except Exception:
					pass
			results.append(p)
	return sorted(results)


def derive_output_paths(input_path: Path) -> dict:
	base = input_path.with_suffix("")
	return {
		"txt": base.with_suffix(".txt"),
		"srt": base.with_suffix(".srt"),
		"vtt": base.with_suffix(".vtt"),
		"csv": base.with_suffix(".segments.csv"),
	}


def write_text(path: Path, text: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8", newline="\n") as f:
		f.write(text.rstrip() + "\n")


def write_srt(path: Path, segments: Iterable) -> None:
	from .subtitles import segments_to_srt

	path.parent.mkdir(parents=True, exist_ok=True)
	srt_text = segments_to_srt(list(segments))
	with path.open("w", encoding="utf-8", newline="\n") as f:
		f.write(srt_text)


def write_vtt(path: Path, segments: Iterable) -> None:
	from .subtitles import segments_to_vtt

	path.parent.mkdir(parents=True, exist_ok=True)
	vtt_text = segments_to_vtt(list(segments))
	with path.open("w", encoding="utf-8", newline="\n") as f:
		f.write(vtt_text)


def write_segments_csv(path: Path, segments: Iterable) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["start", "end", "text"])
		for seg in segments:
			writer.writerow([f"{getattr(seg, 'start', '')}", f"{getattr(seg, 'end', '')}", getattr(seg, 'text', "").strip()])



