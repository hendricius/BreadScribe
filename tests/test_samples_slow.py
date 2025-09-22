import pytest
from pathlib import Path
from typer.testing import CliRunner

from breadscribe.cli import app


SAMPLE_DIR = Path("sample")
FILES = [
	SAMPLE_DIR / "broll_anatoly_stretch_fold_v2.mov",
	SAMPLE_DIR / "broll_hendrik_mixing.mov",
	SAMPLE_DIR / "broll_hendrik_starter_blob.mov",
]


@pytest.mark.slow
def test_samples_with_tiny_en_clean_and_nospeech():
	missing = [p for p in FILES if not p.exists()]
	if missing:
		pytest.skip("sample files not present")

	runner = CliRunner()
	args = [str(SAMPLE_DIR), "--model", "tiny.en", "--workers", "1", "--clean-fillers", "--strict-no-speech"]
	res = runner.invoke(app, args)
	assert res.exit_code == 0

	for p in FILES:
		txt = p.with_suffix(".txt").read_text(encoding="utf-8")
		if "[[no speech detected]]" in txt:
			continue
		# No three identical tokens in a row
		toks = [t.lower() for t in txt.split()]
		bad = any(toks[i] == toks[i-1] == toks[i-2] for i in range(2, len(toks)))
		assert not bad


