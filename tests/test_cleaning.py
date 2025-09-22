from pathlib import Path
import tempfile

from typer.testing import CliRunner

from breadscribe.postprocess import remove_filler_words, remove_filler_from_segments
from breadscribe.cli import app
from breadscribe.types import Segment


def test_remove_filler_words_basic():
	orig = "So, so, okay, like we will now mix the levain."
	clean = remove_filler_words(orig)
	assert clean == "we will now mix the levain"


def test_cli_clean_fillers_flag(monkeypatch):
	runner = CliRunner()
	with tempfile.TemporaryDirectory() as d:
		p = Path(d) / "clip.mov"
		p.write_bytes(b"")

		from breadscribe.postprocess import remove_filler_words as _clean

		def fake_process(*args, **kwargs):
			text = "So, so, okay, like we will now mix the levain."
			# clean_fillers is the 15th positional arg (index 14)
			clean_flag = bool(args[14]) if len(args) > 14 else False
			out_text = _clean(text) if clean_flag else text
			return {
				"skipped": False,
				"path": str(p),
				"rtf": 1.0,
				"detected_language": "en",
				"text": out_text,
			}

		import breadscribe.cli as cli_mod
		monkeypatch.setattr(cli_mod, "_process_single_file", fake_process)

		# Without cleaning
		res1 = runner.invoke(app, [str(p), "--print-text", "--workers", "1"])
		assert res1.exit_code == 0
		assert "So, so, okay, like we will now mix the levain." in res1.stdout

		# With cleaning
		res2 = runner.invoke(app, [str(p), "--print-text", "--workers", "1", "--clean-fillers"])
		assert res2.exit_code == 0
		assert "we will now mix the levain" in res2.stdout
		assert "So, so" not in res2.stdout


def test_collapse_runs():
	orig = "you you you okay okay okay stretch and fold"
	assert "you you you" not in remove_filler_words(orig)
	assert "okay okay okay" not in remove_filler_words(orig)

	segs = [Segment(1, 0.0, 1.0, orig)]
	out = remove_filler_from_segments(segs)
	assert out[0].text.endswith("stretch and fold")


