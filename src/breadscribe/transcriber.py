from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Callable

from faster_whisper import WhisperModel
from .audio_utils import extract_audio, measure_rms_db

from .domain import build_initial_prompt
from .types import Segment, TranscriptResult


 


class Transcriber:
	def __init__(self, model_name: str, device: str, compute_type: str):
		self.model_name = model_name
		self.device = device
		self.compute_type = compute_type
		self._model = None

	def ensure_model(self) -> None:
		# Public method to trigger model initialization early for UX logging
		self._ensure_model()

	def _ensure_model(self) -> WhisperModel:
		if self._model is not None:
			return self._model
		# Smart fallback for device/compute_type
		device = self.device
		compute_type = self.compute_type
		if device == "auto":
			try:
				self._model = WhisperModel(self.model_name, device="cuda", compute_type="float16")
				return self._model
			except Exception:
				self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
				return self._model
		else:
			self._model = WhisperModel(self.model_name, device=device, compute_type=compute_type)
			return self._model

	def transcribe_file(
		self,
		path: Path,
		language: Optional[str],
		initial_prompt: Optional[str],
		beam_size: int,
		temperature: float,
		use_vad: bool,
		vad_params: dict,
		on_segment: Optional[Callable[[Segment], None]] = None,
		audio_stream_index: Optional[int] = None,
		preamp_db: float = 0.0,
		strict_no_speech: bool = True,
		rms_db_threshold: float = -45.0,
		loudnorm: bool = False,
	) -> TranscriptResult:
		model = self._ensure_model()
		prompt = build_initial_prompt(initial_prompt)
		language_arg = None if language == "auto" else language
		media_path = path
		tmp_audio: Optional[Path] = None
		# Always extract audio to allow loudness checks and stream selection
		extracted = extract_audio(path, stream_index=audio_stream_index, gain_db=preamp_db, loudnorm=loudnorm)
		if extracted is not None:
			media_path, used_stream = extracted
			tmp_audio = media_path
		else:
			used_stream = None
		# Near-silence guard: if too quiet, short circuit
		rms_db = measure_rms_db(media_path) if tmp_audio is not None else None
		if strict_no_speech and rms_db is not None and rms_db < rms_db_threshold:
			if tmp_audio is not None:
				try:
					tmp_audio.unlink(missing_ok=True)
				except Exception:
					pass
			return TranscriptResult(detected_language=None, segments=[], text="[[no speech detected]]", rtf=None, used_stream_index=used_stream)
		segments_iter, info = model.transcribe(
			str(media_path),
			language=language_arg,
			beam_size=beam_size,
			temperature=temperature,
			condition_on_previous_text=False,
			vad_filter=use_vad,
			vad_parameters=vad_params,
			initial_prompt=prompt,
		)
		segments: List[Segment] = []
		for i, seg in enumerate(segments_iter):
			# Gate by very short tokens to reduce junk
			text_val = str(getattr(seg, "text", "")).strip()
			if not text_val:
				continue
			letters = [t for t in text_val.split() if any(c.isalpha() for c in t)]
			if len(letters) <= 1 and len("".join(letters)) < 3:
				continue
			segments.append(
				Segment(index=len(segments) + 1, start=float(seg.start), end=float(seg.end), text=text_val)
			)
			if on_segment is not None:
				try:
					on_segment(segments[-1])
				except Exception:
					pass
		text = "\n".join(s.text for s in segments)
		detected_language = None
		try:
			if info is not None and getattr(info, "language", None):
				detected_language = str(info.language)
		except Exception:
			pass
		# naive rtf from info (if available)
		rtf = None
		try:
			dur = float(getattr(info, "duration", 0.0))
			time_spent = float(getattr(info, "transcription_time", 0.0))
			rtf = time_spent / dur if dur > 0 else None
		except Exception:
			pass
		# Cleanup temp
		if tmp_audio is not None:
			try:
				tmp_audio.unlink(missing_ok=True)
			except Exception:
				pass
		return TranscriptResult(detected_language=detected_language, segments=segments, text=text, rtf=rtf, used_stream_index=used_stream)



