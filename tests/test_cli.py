from pathlib import Path
import tempfile

from typer.testing import CliRunner

from breadscribe.cli import app


def test_cli_parsing_and_output_paths(monkeypatch):
	runner = CliRunner()
	with tempfile.TemporaryDirectory() as d:
		p = Path(d) / "clip.mp4"
		p.write_bytes(b"fake")

		# Mock processing to avoid requiring models
		def fake_process(*args, **kwargs):
			return {
				"skipped": False,
				"path": str(p),
				"rtf": 1.0,
				"detected_language": "en",
			}

		import breadscribe.cli as cli_mod
		monkeypatch.setattr(cli_mod, "_process_single_file", fake_process)

		result = runner.invoke(app, [str(p), "--srt", "--vtt", "--segments-csv", "--overwrite"])
		assert result.exit_code == 0
		# Ensure the command ran and printed the summary
		assert "Processed:" in result.stdout



