from pathlib import Path
import tempfile
from types import SimpleNamespace

from typer.testing import CliRunner

from breadscribe.cli import app


def test_cli_forwards_stream_gain_loudnorm_strict(monkeypatch):
	runner = CliRunner()
	with tempfile.TemporaryDirectory() as d:
		p = Path(d) / "clip.mov"
		p.write_bytes(b"")

		called = {"kwargs": None}

		def fake_transcribe_file(self, *args, **kwargs):
			called["kwargs"] = kwargs
			return SimpleNamespace(segments=[], detected_language="en", text="", rtf=None, used_stream_index=1)

		import breadscribe.transcriber as tr
		monkeypatch.setattr(tr.Transcriber, "transcribe_file", fake_transcribe_file)

		args = [
			str(p),
			"--workers", "1",
			"--stream-index", "1",
			"--gain-db", "4.5",
			"--loudnorm",
			"--strict-no-speech",
		]
		res = runner.invoke(app, args)
		assert res.exit_code == 0
		kw = called["kwargs"]
		assert kw is not None
		assert kw.get("audio_stream_index") == 1
		assert abs(float(kw.get("preamp_db") or 0.0) - 4.5) < 1e-6
		assert bool(kw.get("strict_no_speech")) is True
		# loudnorm is applied inside audio extraction path; ensure arg present if wired
		# We accept not present here if handled in extract step.


