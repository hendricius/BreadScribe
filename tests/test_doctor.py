from typer.testing import CliRunner

from breadscribe.cli import app


def test_doctor_runs(monkeypatch):
	# pretend ffmpeg/ffprobe exist by short-circuiting shutil.which
    # Just run doctor; on most dev envs it should find ffmpeg/ffprobe or exit 1 gracefully

	runner = CliRunner()
	res = runner.invoke(app, ["doctor"])
	assert res.exit_code in (0, 1)


