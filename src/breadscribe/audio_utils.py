from __future__ import annotations

import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import json
import math
import struct
from .errors import DependencyNotFound, AudioExtractionError, ProbeError

def _require(cmd: str) -> None:
	if shutil.which(cmd) is None:
		raise DependencyNotFound(f"Required tool not on PATH: {cmd}")

def extract_audio(
    input_path: Path,
    stream_index: Optional[int] = None,
    gain_db: float = 0.0,
    loudnorm: bool = False,
) -> Optional[Tuple[Path, Optional[int]]]:
    """Extract audio to a temporary mono 16k WAV using ffmpeg.

    - If stream_index is None, choose best stream via ffprobe.
    - Applies mild HP/LP filter, optional loudnorm and gain.
    Returns temp path or None on failure.
    """
    try:
        _require("ffmpeg")
        tmpf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = Path(tmpf.name)
        tmpf.close()
        # Pick stream automatically if not specified
        idx = stream_index
        if idx is None:
            try:
                streams = ffprobe_streams(input_path)
                idx = choose_best_audio_stream(streams)
            except Exception:
                idx = None
        idx_map = ["-map", f"0:a:{idx}?" ] if idx is not None else ["-map", "a:0?"]

        afilter: list[str] = ["highpass=f=60", "lowpass=f=8000"]
        if loudnorm:
            afilter.append("loudnorm=I=-23:TP=-2:LRA=7:dual_mono=true")
        if gain_db and abs(gain_db) > 0.01:
            afilter.append(f"volume={gain_db}dB")
        agraph = ",".join(afilter)

        cmd = [
            "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
            "-y",
            "-i", str(input_path),
            "-vn", "-sn", "-dn",
            *idx_map,
            "-ac", "1", "-ar", "16000",
            "-af", agraph,
            str(tmp_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", "ignore") if e.stderr else ""
            raise AudioExtractionError(stderr.strip() or "ffmpeg failed") from e
        return tmp_path, idx
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)  # type: ignore[name-defined]
        except Exception:
            pass
        return None



def ffprobe_streams(input_path: Path) -> list[dict]:
	"""Return audio stream metadata via ffprobe as a list of dicts, or []."""
	_require("ffprobe")
	cmd = [
		"ffprobe", "-v", "error",
		"-select_streams", "a",
		"-show_entries", "stream=index,codec_name,channels,bit_rate,channel_layout,language,tags",
		"-of", "json", str(input_path),
	]
	try:
		out = subprocess.run(cmd, check=True, capture_output=True).stdout
		data = json.loads(out.decode("utf-8", errors="ignore"))
		return data.get("streams", [])
	except subprocess.CalledProcessError as e:
		raise ProbeError(e.stderr.decode("utf-8", "ignore") if e.stderr else "ffprobe failed") from e
	except json.JSONDecodeError as e:
		raise ProbeError("ffprobe returned invalid JSON") from e


def choose_best_audio_stream(streams: list[dict]) -> Optional[int]:
	"""Choose best stream: prefer English language, more channels, higher bitrate."""
	if not streams:
		return None

	def lang_score(s: dict) -> int:
		tags = s.get("tags") or {}
		lang = (tags.get("language") or s.get("language") or "").lower()
		return 1 if lang.startswith("en") else 0

	def bit_rate(s: dict) -> int:
		try:
			return int(s.get("bit_rate") or 0)
		except Exception:
			return 0

	ranked = sorted(
		streams,
		key=lambda s: (lang_score(s), int(s.get("channels") or 0), bit_rate(s)),
		reverse=True,
	)
	try:
		return int(ranked[0].get("index"))
	except Exception:
		return None


def measure_rms_db(wav_path: Path) -> Optional[float]:
	"""Compute RMS in dBFS for 16-bit mono WAV written by extract_audio."""
	try:
		with wav_path.open("rb") as f:
			_ = f.read(44)
			data = f.read()
		samples = struct.iter_unpack("<h", data)
		total = 0.0
		count = 0
		peak = 32768.0
		for (s,) in samples:
			total += (s / peak) ** 2
			count += 1
		if count == 0:
			return None
		rms = math.sqrt(total / count)
		rms = max(rms, 1e-12)
		return 20.0 * math.log10(rms)
	except Exception:
		return None

