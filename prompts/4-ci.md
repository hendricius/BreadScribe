## Goal
Polish the repository for GitHub. Add an automatic test workflow that runs on every push to `main` and on every pull request, and refresh the README to focus on BreadScribe as a simple tool for transcribing bread related videos. Lean on the existing Makefile so that both users and CI call `make` targets.

Apply the following changes exactly.

---

## 1) Create or replace `.github/workflows/ci.yml`

This workflow installs the project with test and lint extras, then runs lint and tests on Ubuntu for Python 3.10 to 3.12. It triggers on pushes to `main` and on all pull requests. It skips slow tests by default. It builds the package to validate packaging metadata.

**File:** `.github/workflows/ci.yml`
```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test:
    name: Tests py${{ matrix.python }}
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python: ["3.10", "3.11", "3.12"]

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
          cache: pip

      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[test,lint]"

      - name: Lint
        run: |
          ruff check .

      - name: Unit tests
        run: |
          # Skip slow tests in CI. They can run locally or in a manual workflow.
          pytest -q -m "not slow" --cov=breadscribe --cov-report=term-missing

      - name: Build sdist and wheel
        run: |
          python -m pip install build twine
          python -m build
          python -m twine check dist/*
Optional: if you want a manual workflow to run slow sample tests, add .github/workflows/slow.yml later. Not required for this prompt.

2) Update the Makefile
Ensure the Makefile has these targets, since both the README and CI rely on them. If a target already exists, keep its behavior or align it with the definitions below.

File: Makefile

make
Copy code
.PHONY: setup lint test test-sample transcribe-sample run clean

# Install project with test and lint extras
setup:
	python -m pip install --upgrade pip
	pip install -e ".[test,lint]"

# Lint only
lint:
	ruff check .

# Fast tests only, skip slow tests that use real media
test:
	pytest -q -m "not slow" --cov=breadscribe --cov-report=term-missing

# Run the slow suite on local machines. Requires sample media in sample/
test-sample:
	pytest -q -m slow

# Transcribe the sample clips with safe defaults
transcribe-sample:
	breadscribe sample --model tiny.en --workers 1 --clean-fillers --strict-no-speech --srt --vtt || true
	@echo "---- OUTPUTS ----"; for f in sample/*.txt; do echo $$f; sed -n '1,12p' $$f; echo ""; done

# Run BreadScribe with custom args, example:
# make run ARGS="sample --srt"
run:
	breadscribe $(ARGS)

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build *.egg-info
	find . -name "*.pyc" -delete
3) Replace the top of README.md
Replace everything in README.md with the content below. Keep license and badges if you have additional ones. Update any project links if your repository slug differs.

File: README.md

markdown
Copy code
# BreadScribe

[![CI](https://github.com/hendricius/breadscriber/actions/workflows/ci.yml/badge.svg)](https://github.com/hendricius/breadscriber/actions/workflows/ci.yml)

BreadScribe is your tool to transcribe bread related videos, especially sourdough focused clips. Pass it one or many media files, and it writes a plain text file next to each video with the same base name. You can also opt in to SRT or VTT subtitles.

BreadScribe includes a small dictionary of sourdough terms from my book **The Sourdough Framework**, which helps the model keep domain words like levain, autolyse, and bulk fermentation intact.

## Features

- Local transcription with open source models, no paid APIs
- Works on single files or folders
- Sidecar outputs next to each input: `clip.mp4` becomes `clip.txt`
- Optional `--srt` and `--vtt` subtitle files
- Cleaning pass to reduce filler phrases and collapsed word repeats
- Audio stream selection and near silence guard for b roll

## Requirements

- Python 3.10 or newer
- FFmpeg tools on PATH: `ffmpeg` and `ffprobe`
  - macOS: `brew install ffmpeg`
  - Ubuntu: `sudo apt-get update && sudo apt-get install -y ffmpeg`
  - Windows: download FFmpeg release, add the `bin` folder to PATH

## Install

Create a virtual environment if you prefer, then install the project in editable mode with test and lint tools.

```bash
python -m pip install --upgrade pip
pip install -e ".[test,lint]"
If you like to keep workflows simple, you can use the Makefile targets below.

Quick start
Use the Makefile for common tasks.

bash
Copy code
# One time setup, installs dependencies
make setup

# Transcribe the bundled sample videos with safe defaults
make transcribe-sample

# Run fast tests
make test

# Run the slow sample tests locally only
make test-sample

# Transcribe your own folder, write SRT too
make run ARGS="/path/to/videos --srt"

# Transcribe a single file with extra cleaning
make run ARGS="sample/broll_hendrik_mixing.mov --clean-fillers --strict-no-speech"
Basic CLI usage
Although the Makefile is convenient, you can call the CLI directly.

bash
Copy code
breadscribe /path/to/folder
breadscribe /path/to/clip.mp4 --srt
breadscribe /clips --model large-v3 --device cuda --compute-type float16
breadscribe /clips --language auto --clean-fillers
Notes on the sourdough glossary
The tool ships with a small list of common sourdough terms. You can bias transcription further with --initial-prompt "your, extra, terms" when needed.

Troubleshooting
ffmpeg: command not found: install FFmpeg and ensure it is on PATH.

Output is junk for b roll: try --strict-no-speech and --clean-fillers. Use --stream-index 0 or 1 if the file has multiple audio tracks.

Slow on CPU: use distil-large-v3 or tiny.en. On NVIDIA GPUs use --device cuda --compute-type float16.

Contributing
Issues and pull requests are welcome. Please run make lint and make test before you open a PR.

License
MIT

yaml
Copy code

---

## 4) Verify `pyproject.toml` extras

Make sure the optional dependencies referenced by CI and the Makefile exist. If the section is missing, add it.

**File:** `pyproject.toml` additions
```toml
[project.optional-dependencies]
test = ["pytest>=8.2", "pytest-cov>=5.0", "pytest-mock>=3.14"]
lint = ["ruff>=0.6.3"]
Also ensure the console script entry points expose the CLI.

toml
Copy code
[project.scripts]
breadscribe = "breadscribe.cli:app"
breadscriber = "breadscribe.cli:app"
5) Acceptance criteria for this change
Cursor should validate the following before completing the task.

The CI workflow is present at .github/workflows/ci.yml.

A pull request in this repository triggers the workflow and runs tests.

make setup, make test, and make transcribe-sample work locally.

README.md shows the new description, the badge, the Makefile based quick start, and accurate requirements.

pyproject.toml contains test and lint extras, and both breadscribe and breadscriber entry points.

If any step fails, continue updating until all checks pass.