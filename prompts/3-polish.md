Purpose: polish Breadscriber for open source release on GitHub, improve reliability, make errors actionable, and add strong CI. The items are grouped as TODOs with concrete actions and code-ready snippets. You can assign issues straight from these headings.

---

## 0) Naming and repo hygiene

**Problem**
- The repository name and Python package name appear inconsistent, for example repo `breadscriber` and package `breadscribe`. This confuses users and tooling.

**Actions**
- Decide on the public name. Recommend:
  - Keep the **project** and **repo** name: `breadscriber`.
  - Keep the **Python package** name as `breadscribe` for now, then migrate at 1.0.0.
- Add an alias CLI entry point so both `breadscribe` and `breadscriber` work.

**Snippets**
`pyproject.toml`:
```toml
[project.scripts]
breadscribe = "breadscribe.cli:app"
breadscriber = "breadscribe.cli:app"
Docs

Update README title and badges to “Breadscriber”.

Explain import and CLI aliases clearly.

1) Exception taxonomy and error handling
Problem

Several except Exception: blocks hide root causes. Users see generic “Error” without remediation steps.

Subprocess errors from ffmpeg and JSON decoding from ffprobe deserve specific messages and exit codes.

Actions

Introduce a small error hierarchy in src/breadscribe/errors.py.

Replace broad except Exception with targeted exceptions.

At the CLI boundary, catch only BreadscribeError and print friendly help; let unexpected exceptions fail the process with a nonzero code and a short trace.

Add src/breadscribe/errors.py:

python
Copy code
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
Refactor audio_utils.py where you run external tools:

python
Copy code
import shutil
from .errors import DependencyNotFound, AudioExtractionError, ProbeError

def _require(cmd: str) -> None:
    if shutil.which(cmd) is None:
        raise DependencyNotFound(f"Required tool not on PATH: {cmd}")

def ffprobe_streams(input_path: Path) -> list[dict]:
    _require("ffprobe")
    try:
        # existing call
    except subprocess.CalledProcessError as e:
        raise ProbeError(e.stderr.decode("utf-8", "ignore")) from e
    except json.JSONDecodeError as e:
        raise ProbeError("ffprobe returned invalid JSON") from e
Refactor extract_audio(...):

python
Copy code
_require("ffmpeg")
try:
    # existing ffmpeg command
except subprocess.CalledProcessError as e:
    stderr = e.stderr.decode("utf-8", "ignore") if e.stderr else ""
    raise AudioExtractionError(stderr.strip() or "ffmpeg failed") from e
Refactor transcriber.py:

Wrap model load failures as TranscriptionError.

When RMS is below threshold and strict_no_speech is true, raise NoSpeechDetected so the CLI can decide output text and exit code.

CLI boundary cli.py:

python
Copy code
from .errors import BreadscribeError, NoSpeechDetected, DependencyNotFound

try:
    result = _process_single_file(...)
except NoSpeechDetected:
    # write [[no speech detected]] and continue to next file
    ...
except DependencyNotFound as e:
    console.print(f"[red]Dependency missing:[/red] {e}. See README install section.")
    raise typer.Exit(code=2)
except BreadscribeError as e:
    console.print(f"[red]Error:[/red] {e}")
    continue
2) Do not duplicate shared types
Problem

Segment is defined in both transcriber.py and subtitles.py. This invites drift.

Actions

Create src/breadscribe/types.py with Segment and TranscriptResult.

Import these in transcriber.py, subtitles.py, io_utils.py, postprocess.py, and tests.

Snippet

python
Copy code
# src/breadscribe/types.py
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Segment:
    index: int
    start: float
    end: float
    text: str

@dataclass
class TranscriptResult:
    detected_language: Optional[str]
    segments: List[Segment]
    text: str
    rtf: Optional[float]
    used_stream_index: Optional[int] = None
3) Dependency checks and doctor command
Problem

Users often lack ffmpeg or ffprobe. Failures are cryptic.

Actions

Add a breadscribe doctor subcommand that checks:

Python version

faster-whisper import

ffmpeg and ffprobe availability and versions

CUDA availability when requested

Exit codes: 0 when OK, 1 when any check fails.

Snippet in cli.py:

python
Copy code
@app.command()
def doctor() -> None:
    import platform, shutil, subprocess
    ok = True
    def check(cmd):
        nonlocal ok
        if shutil.which(cmd) is None:
            console.print(f"[red]{cmd} not found[/red]")
            ok = False
        else:
            out = subprocess.run([cmd, "-version"], capture_output=True, text=True)
            console.print(f"[green]{cmd}[/green]: {out.stdout.splitlines()[0]}")
    console.print(f"Python: {platform.python_version()}")
    check("ffmpeg")
    check("ffprobe")
    try:
        import faster_whisper
        console.print(f"faster-whisper: {faster_whisper.__version__}")
    except Exception as e:
        console.print(f"[red]faster-whisper import failed:[/red] {e}")
        ok = False
    raise typer.Exit(code=0 if ok else 1)
4) Logging and verbosity
Problem

Some diagnostic info is printed, but there is no structured verbosity control, which makes debugging user reports harder.

Actions

Add --verbose flag that prints chosen audio stream metadata, RMS dB, language, and RTF.

Route diagnostics through rich with levels.

Snippet

python
Copy code
verbose: bool = typer.Option(False, help="Print diagnostics")
...
if verbose:
    console.print(Panel.fit(f"Stream index: {res.get('used_stream_index')}\nRMS dBFS: {rms:.1f}\nLanguage: {lang}"))
5) Dutch friendly defaults and glossary
Problem

You mentioned Dutch. If your channel is mostly Dutch, set smart defaults and an extended glossary.

Actions

Add src/breadscribe/glossary_nl.txt with Dutch baking terms.

Add --glossary option: auto, en, nl. When language="auto", pick glossary based on detected language for the next files in the same run.

Snippet in domain.py:

python
Copy code
def load_glossary(lang: str = "en") -> list[str]:
    name = "glossary_nl.txt" if lang.lower().startswith("nl") else "glossary.txt"
    with importlib_resources.files("breadscribe").joinpath(name).open(encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]
6) Stream selection transparency
Problem

Audio stream selection is automatic, but the user cannot see what was chosen.

Actions

In transcriber.transcribe_file, return used_stream_index and stream tags.

CLI prints this when --verbose is enabled.

Add --stream-index override and document it in README.

7) Postprocessing improvements
Problem

Repetition and filler removal works, but can be more robust.

Actions

Extend postprocess.remove_filler_words to collapse punctuation runs and remove bracketed background annotations like [music], (laughs).

Add property based tests for the normalizer, see Section 10.

Snippet

python
Copy code
_BRACKET_RE = re.compile(r"[\[\(][^\]\)]{0,30}[\]\)]")
t = _BRACKET_RE.sub("", t)
t = re.sub(r"[ ]{2,}", " ", t)
8) File discovery and safety
Problem

Discovery may include hidden files or partial downloads. Writers assume paths exist.

Actions

Skip hidden files and files smaller than a few kilobytes by default, allow override with --include-hidden.

Ensure all writers create parent directories and normalize newlines.

Snippet

python
Copy code
def discover_inputs(path: Path, recursive: bool, allowed_exts: set[str], include_hidden: bool=False) -> list[Path]:
    ...
    if not include_hidden and any(part.startswith(".") for part in p.relative_to(path).parts):
        continue
9) Concurrency
Problem

Global model reuse across processes must be explicit for Windows spawn semantics. Partial globals can break on Windows.

Actions

Use ProcessPoolExecutor with an initializer and initargs to load the model once per worker.

Keep default workers to 1 on all platforms; document memory impact when users increase.

Snippet

python
Copy code
def _init_worker(model_name: str, device: str, compute_type: str):
    global _GLOBAL_TRANSCRIBER
    _GLOBAL_TRANSCRIBER = Transcriber(model_name=model_name, device=device, compute_type=compute_type)

def _get_transcriber() -> Transcriber:
    return _GLOBAL_TRANSCRIBER
10) Static analysis, formatting, and pre-commit
Problem

Style and typing vary across modules. Reviewers value consistent tooling.

Actions

Add Ruff for linting and formatting, MyPy for basic type checking, and a pre-commit config.

Files
.pre-commit-config.yaml:

yaml
Copy code
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.3
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: check-yaml
      - id: end-of-file-fixer
      - id: trailing-whitespace
pyproject.toml additions:

toml
Copy code
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.mypy]
python_version = "3.10"
warn_unused_ignores = true
check_untyped_defs = true
ignore_missing_imports = true
11) Tests, coverage, and sample assets policy
Problem

Tests exist, good, but add more guardrails and keep CI fast.

Actions

Add tests for:

doctor command exit codes

CLI exit codes for missing ffmpeg and for --stream-index out of range

RMS gate thresholds

Error translations to BreadscribeError

Add property based tests for normalizer using Hypothesis.

Snippets
pyproject.toml:

toml
Copy code
[project.optional-dependencies]
test = ["pytest>=8.2", "pytest-cov>=5.0", "pytest-mock>=3.14", "hypothesis>=6.112"]
lint = ["ruff>=0.6.3", "mypy>=1.11"]
CI policy

Keep real sample transcription tests under @pytest.mark.slow.

Skip slow tests in CI by default. Provide a separate workflow that can be triggered manually to run slow tests on GitHub if you store tiny public samples or use Git LFS.

12) CI workflows
Problem

Current CI runs tests on Ubuntu only without lint or cache.

Actions

Replace with a matrix for Python 3.10 to 3.12 on Ubuntu, add Windows minimal smoke, add caching and lint, run unit tests only, and verify the package builds.

Add manual workflow slow.yml that runs pytest -m slow if samples are present.

Optionally, add release workflow that publishes to PyPI on a tag after tests pass.

Files
.github/workflows/ci.yml:

yaml
Copy code
name: CI

on:
  push:
    branches: ["**"]
  pull_request:

jobs:
  test:
    name: Tests ${{ matrix.os }} py${{ matrix.python }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest]
        python: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
          cache: "pip"
      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[test,lint]"
      - name: Lint
        run: |
          ruff check .
      - name: Type check
        run: |
          mypy src/breadscribe || true
      - name: Unit tests
        run: |
          pytest -q --cov=breadscribe --cov-report=xml -m "not slow"
      - name: Build sdist and wheel
        run: |
          python -m pip install build twine
          python -m build
          python -m twine check dist/*

  smoke-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"
      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[test]"
      - name: Smoke tests
        run: |
          pytest -q -k "subtitles or cleaning" -m "not slow"
Optional manual slow suite .github/workflows/slow.yml:

yaml
Copy code
name: Slow tests

on:
  workflow_dispatch:

jobs:
  slow:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
          cache: "pip"
      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[test]"
      - name: Run slow tests if samples exist
        run: |
          if ls sample/*.mov 1> /dev/null 2>&1; then
            pytest -q -m slow
          else
            echo "No sample files present, skipping."
          fi
13) Packaging, metadata, and docs polish
Problem

Open source consumers expect complete metadata and contribution docs.

Actions

Enrich pyproject.toml with:

urls block for Homepage, Repository, Issues, Changelog.

keywords, authors with your name and email, maintainers.

Add CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md, CHANGELOG.md.

Add badges to README for PyPI, CI, coverage.

Add a Troubleshooting section:

ffmpeg not found on macOS, Windows, Linux

garbled text hints, use --stream-index, --loudnorm, --clean-fillers

no speech threshold knob

Snippet
pyproject.toml:

toml
Copy code
[project]
keywords = ["speech-to-text", "whisper", "asr", "subtitle", "ffmpeg", "nl", "en"]
authors = [{ name = "Hendrik", email = "your@email" }]
maintainers = [{ name = "Hendrik" }]
urls = { Homepage = "https://github.com/hendricius/breadscriber",
         Repository = "https://github.com/hendricius/breadscriber",
         Issues = "https://github.com/hendricius/breadscriber/issues",
         Changelog = "https://github.com/hendricius/breadscriber/releases" }
14) Release process
Problem

No defined versioning or release automation.

Actions

Adopt SemVer. Keep 0.y while iterating, cut 1.0.0 when API is stable.

Add a release workflow that publishes to PyPI on tag v*.

Generate a GitHub Release with notes pulled from CHANGELOG.md.

Optional workflow .github/workflows/release.yml:

yaml
Copy code
name: Release

on:
  push:
    tags: ["v*"]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
          cache: "pip"
      - name: Build
        run: |
          python -m pip install --upgrade pip build twine
          python -m build
          python -m twine check dist/*
      # Configure PYPI_API_TOKEN secret in repo settings
      - name: Publish to PyPI
        env:
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
          TWINE_USERNAME: __token__
        run: |
          python -m twine upload dist/*
15) Small quality fixes
Normalize newlines to \n in all writers. You already write UTF 8, keep it.

Use pathlib.Path consistently in tests.

Ensure derive_output_paths is used everywhere to prevent mismatched extensions.

Print a single detected language line per file even when --verbose is off.

Add --version option to print package version:

python
Copy code
@app.callback()
def _version_callback(version: bool = typer.Option(False, "--version", help="Show version", is_eager=True)):
    if version:
        import importlib.metadata as md
        typer.echo(md.version("breadscribe"))
        raise typer.Exit()
16) Contributor experience
Add issue templates for bug report and feature request.

Add a PR template with checkboxes:

tests added

docs updated

lint passing

Add “good first issue” labels to a few items like doctor command, glossary nl.

17) Final verification checklist
 Exceptions refactored to specific classes, CLI maps them to clear messages and exit codes.

 Shared types moved to types.py, imports updated.

 doctor command added.

 --verbose, --stream-index, and Dutch glossary switch documented and tested.

 Pre-commit, Ruff, and optional MyPy wired into CI.

 Matrix CI green on Ubuntu for py3.10 to 3.12, Windows smoke green.

 README, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, CHANGELOG present.

 Release workflow added or deferred until after first public feedback.

 Tag v0.2.0 for this round of changes, cut a GitHub Release.

18) Optional ideas for later
Add JSONL export for downstream AI tooling.

Add --segments-json with logprob and no speech probability where available.

Provide a small breadscribe doctor --sample that extracts and prints the first 3 seconds waveform RMS, helpful for bug reports.
