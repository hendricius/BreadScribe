class BreadscribeError(Exception):
	"""Base class for expected, user facing errors."""


class DependencyNotFound(BreadscribeError):
	"""ffmpeg or ffprobe missing."""


class AudioExtractionError(BreadscribeError):
	"""ffmpeg failed to extract audio."""


class ProbeError(BreadscribeError):
	"""ffprobe failed or returned malformed data."""


class NoSpeechDetected(BreadscribeError):
	"""RMS gate says near silence."""


class TranscriptionError(BreadscribeError):
	"""faster-whisper raised during decode."""


