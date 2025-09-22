from pathlib import Path
import tempfile
from types import SimpleNamespace

from typer.testing import CliRunner

from breadscribe.cli import app


def test_cli_prints_language_and_stream(monkeypatch):
	runner = CliRunner()
	with tempfile.TemporaryDirectory() as d:
		p = Path(d) / "clip.mov"
		p.write_bytes(b"")

		def fake_transcribe_file(self, *args, **kwargs):
			return SimpleNamespace(segments=[], detected_language="en", text="[[no speech detected]]", rtf=None, used_stream_index=0)

		import breadscribe.transcriber as tr
		monkeypatch.setattr(tr.Transcriber, "transcribe_file", fake_transcribe_file)

		res = runner.invoke(app, [str(p), "--workers", "1", "--print-text"])
		assert res.exit_code == 0
		# CLI should announce language and stream index
		assert "Detected language: en" in res.stdout
		assert "stream=0" in res.stdout or "Stream: 0" in res.stdout


