from pathlib import Path
import tempfile

from typer.testing import CliRunner

from breadscribe.cli import app


def test_print_text_processed(monkeypatch):
	runner = CliRunner()
	with tempfile.TemporaryDirectory() as d:
		p = Path(d) / "a.mov"
		p.write_bytes(b"")

		def fake_process(*args, **kwargs):
			return {
				"skipped": False,
				"path": str(p),
				"rtf": 0.5,
				"detected_language": "en",
				"text": "Hello world",
			}

		import breadscribe.cli as cli_mod
		monkeypatch.setattr(cli_mod, "_process_single_file", fake_process)

		res = runner.invoke(app, [str(p), "--print-text", "--workers", "1"])
		assert res.exit_code == 0
		assert "Hello world" in res.stdout


def test_print_text_skipped(monkeypatch):
	runner = CliRunner()
	with tempfile.TemporaryDirectory() as d:
		p = Path(d) / "b.mov"
		p.write_bytes(b"")

		def fake_process(*args, **kwargs):
			return {
				"skipped": True,
				"path": str(p),
				"rtf": None,
				"detected_language": None,
				"text": "Existing transcript text",
			}

		import breadscribe.cli as cli_mod
		monkeypatch.setattr(cli_mod, "_process_single_file", fake_process)

		res = runner.invoke(app, [str(p), "--print-text", "--workers", "1"])
		assert res.exit_code == 0
		assert "Existing transcript text" in res.stdout


def test_progress_shows_current_file(monkeypatch):
	runner = CliRunner()
	with tempfile.TemporaryDirectory() as d:
		p1 = Path(d) / "c1.mov"
		p2 = Path(d) / "c2.mov"
		p1.write_bytes(b"")
		p2.write_bytes(b"")

		def fake_process(*args, **kwargs):
			return {
				"skipped": False,
				"path": str(args[0]),
				"rtf": 1.0,
				"detected_language": "en",
				"text": "X",
			}

		import breadscribe.cli as cli_mod
		monkeypatch.setattr(cli_mod, "_process_single_file", fake_process)

		res = runner.invoke(app, [str(Path(d)), "--workers", "1", "--print-text"])
		assert res.exit_code == 0
		# Should include per-file step text with names and counts
		assert "Decoding and transcribing:" in res.stdout
		assert "Step 2/3 [1/2]" in res.stdout
		assert "Writing outputs:" in res.stdout

