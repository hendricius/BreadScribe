from pathlib import Path
import tempfile

from typer.testing import CliRunner

from breadscribe.cli import app


def test_cli_skips_existing_without_overwrite(monkeypatch):
	runner = CliRunner()
	with tempfile.TemporaryDirectory() as d:
		p = Path(d) / "clip.mov"
		p.write_bytes(b"")
		txt = p.with_suffix(".txt")
		txt.write_text("OLD", encoding="utf-8")

		# Ensure _process_single_file is not called by raising if it is
		import breadscribe.cli as cli_mod
		called = {"v": False}

		def fake_process(*args, **kwargs):
			called["v"] = True
			return {"path": str(p), "rtf": None, "detected_language": None, "text": "NEW"}

		monkeypatch.setattr(cli_mod, "_process_single_file", fake_process)

		res = runner.invoke(app, [str(p), "--workers", "1"])  # no --overwrite
		assert res.exit_code == 0
		assert txt.read_text(encoding="utf-8") == "OLD"
		assert called["v"] is False


def test_cli_overwrites_with_flag(monkeypatch):
	runner = CliRunner()
	with tempfile.TemporaryDirectory() as d:
		p = Path(d) / "clip.mov"
		p.write_bytes(b"")
		txt = p.with_suffix(".txt")
		txt.write_text("OLD", encoding="utf-8")

		import breadscribe.cli as cli_mod
		
		def fake_process(*args, **kwargs):
			return {"path": str(p), "rtf": None, "detected_language": "en", "text": "NEW"}

		monkeypatch.setattr(cli_mod, "_process_single_file", fake_process)

		res = runner.invoke(app, [str(p), "--workers", "1", "--overwrite", "--print-text"])
		assert res.exit_code == 0
		assert txt.read_text(encoding="utf-8").strip() == "NEW"
		assert "NEW" in res.stdout


