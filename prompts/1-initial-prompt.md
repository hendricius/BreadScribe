## Project title
Breadscribe, a local video to transcript CLI

## Goal
Build a cross platform Python 3.10 plus command line tool that takes a file or folder of media, transcribes the audio locally with open source models, and writes a plain text file next to each input with the same base name. Example: `my_clip.mp4` becomes `my_clip.txt`. Add optional SRT and VTT subtitle outputs. No paid APIs.

## Constraints
- Must run fully local, no network calls at runtime for inference.
- Must rely on open source tools and models.
- GPU acceleration is optional, the tool must run well on CPU.
- Keep installation simple for macOS, Windows, and Linux.
- Default behavior: produce `basename.txt` with the segment texts concatenated in order.

## Primary approach and libraries
- Use `faster-whisper` for transcription. It provides a CTranslate2 based implementation of Whisper that is fast and memory efficient.
- Use the `WhisperModel` API to load models by name: support `tiny`, `base`, `small`, `medium`, `large-v1`, `large-v2`, `large-v3`, and `distil-large-v2`.
- Default model: `distil-large-v2` for good quality on CPU. Allow the user to switch to `large-v3` on GPU for maximum accuracy.
- Use built in VAD filtering from faster-whisper to skip silence by default.
- Use PyAV via faster-whisper to decode media from path. If FFmpeg is required on the system, document how to install it in the README.

## Command line interface
Create a CLI named `breadscribe` using `typer` with rich progress output.

**Invocation examples**
- `breadscribe /path/to/video_or_folder`
- `breadscribe /path/to/folder --srt --vtt --recursive`
- `breadscribe clip.mp4 --model large-v3 --device cuda --compute-type float16`
- `breadscribe /clips --language en --initial-prompt "whole wheat, sourdough, autolyse"`

**Flags**
- `--model` one of: `distil-large-v2`, `large-v3`, `medium`, `small`, `base`, default `distil-large-v2`
- `--device` `auto`, `cuda`, `cpu`, default `auto` which selects CUDA if available
- `--compute-type` `auto`, `float16`, `int8_float16`, `int8`, default `auto` which picks `float16` on CUDA and `int8` on CPU
- `--language` ISO code or `auto`, default `en`
- `--initial-prompt` free text biasing prompt, appended to a built in baking glossary
- `--srt` also write `basename.srt`
- `--vtt` also write `basename.vtt`
- `--segments-csv` also write `basename.segments.csv` with start, end, text
- `--recursive` recurse into subfolders
- `--overwrite` overwrite existing outputs
- `--workers` integer, number of parallel worker processes at file level, default `1`
- `--no-vad` disable VAD filtering

**Output rules**
- Always write `basename.txt` as UTF 8 text, one line per segment, in order.
- When `--srt` or `--vtt` is specified, write well formatted subtitle files with timestamps.
- When `--segments-csv` is specified, write a CSV with headers: `start,end,text`.

## Quality and stability settings
Use stable defaults, make them configurable through the CLI.
- `temperature=0.0`
- `beam_size=5`
- `condition_on_previous_text=False`
- `vad_filter=True` unless `--no-vad` is set
- `vad_parameters=dict(min_silence_duration_ms=500)`
- If `--language auto`, detect language and print it once per file.

## Domain prompt for baking terms
Include a small baked in glossary to reduce mishears of domain terms. Merge it with the user supplied `--initial-prompt`.

Suggested glossary terms:
`autolyse, levain, poolish, preferment, bulk fermentation, crumb, oven spring, banneton, hydration, baker’s percent, lamination, stretch and fold, slap and fold, scoring, couche, pâte fermentée, tangzhong, yudane, retard, aliquot jar, levain build, shaping, batard, boule, preheat, steam, Dutch oven, crumb shot, to 96 F internal`

## File discovery
- Accept extensions: `.mp4, .mov, .mkv, .m4v, .avi, .flv, .mp3, .wav, .m4a, .aac, .ogg, .flac`
- For a folder input, scan eligible files, optionally recursively with `--recursive`
- Skip inputs that already have a `.txt` next to them unless `--overwrite` is set

## Performance and memory
- Initialize the model once per process, then reuse it for each file that process handles.
- On CUDA: default `compute_type=float16`. On CPU: default `compute_type=int8`.
- Use process level parallelism for multiple files when `--workers > 1`. Keep the default at `1` to avoid loading multiple models for users with limited RAM.
- Print average real time factor at the end as a simple performance summary.

## Extensibility plan
- Keep diarization out of scope for v1. Leave an extension point for a future `--diarize` flag that could integrate WhisperX.
- Provide a clean data structure for segments so future JSON or JSONL export is trivial.
- Keep the transcriber decoupled from IO to allow unit testing without heavy models.

## Project structure
Create the following layout.

```

breadscribe/
pyproject.toml
README.md
LICENSE
src/breadscribe/**init**.py
src/breadscribe/cli.py
src/breadscribe/transcriber.py
src/breadscribe/io\_utils.py
src/breadscribe/subtitles.py
src/breadscribe/domain.py
tests/test\_cli.py
tests/test\_subtitles.py
.github/workflows/ci.yml

```

## Implementation details

### `src/breadscribe/transcriber.py`
- Implement `class Transcriber` that manages model lifetime.
- Constructor args: `model_name: str`, `device: str`, `compute_type: str`.
- Build `WhisperModel(model_name, device=..., compute_type=...)` with a smart fallback:
  - If `device="auto"`, try CUDA with `float16`, on failure fall back to CPU with `int8`.
- `transcribe_file(path: Path, language: Optional[str], initial_prompt: Optional[str], beam_size: int, temperature: float, use_vad: bool, vad_params: dict) -> TranscriptResult`
- `TranscriptResult` data class: `detected_language: Optional[str]`, `segments: list[Segment]`, `text: str`
- `Segment` data class: `index: int`, `start: float`, `end: float`, `text: str`
- Call `model.transcribe` with the defaults listed above, pass `initial_prompt` produced by `build_initial_prompt`.
- Collect segments, compose `text` by joining segment texts with newlines, return `TranscriptResult`.

### `src/breadscribe/subtitles.py`
- Render functions:
  - `segments_to_srt(segments: list[Segment]) -> str`
  - `segments_to_vtt(segments: list[Segment]) -> str`
- Timestamp formatting:
  - SRT `HH:MM:SS,mmm`
  - VTT `HH:MM:SS.mmm`
- Ensure proper numbering and blank lines between SRT cues.

### `src/breadscribe/io_utils.py`
- Functions:
  - `discover_inputs(path: Path, recursive: bool, allowed_exts: set[str]) -> list[Path]`
  - `derive_output_paths(input_path: Path) -> dict` with keys `txt`, `srt`, `vtt`, `csv`
  - `write_text(path: Path, text: str)`
  - `write_srt(path: Path, segments: list[Segment])`
  - `write_vtt(path: Path, segments: list[Segment])`
  - `write_segments_csv(path: Path, segments: list[Segment])`
- Always write UTF 8 with newline normalization.

### `src/breadscribe/domain.py`
- `BAKING_GLOSSARY: list[str]`
- `build_initial_prompt(user_prompt: Optional[str]) -> str` that merges the glossary with the user prompt into a natural sentence, for example:
  - `This video may contain baking terms such as autolyse, levain, poolish, preferment, hydration, banneton, crumb, oven spring.`

### `src/breadscribe/cli.py`
- Use `typer` to define arguments and options that match the CLI spec.
- Show a progress bar over files with `rich`.
- Detect language once per file when `language="auto"`, print a single line like:
  - `Detected language: en`
- Respect `--overwrite`. Skip files that already have `.txt` when not overwriting.
- On exceptions, log an error and continue to the next file, do not crash the whole batch.
- Optional file level parallelism using `concurrent.futures.ProcessPoolExecutor` when `--workers > 1`.

## Dependencies in `pyproject.toml`
- Python `>=3.10`
- `faster-whisper>=1.0.0`
- `typer[all]>=0.12.0`
- `rich>=13.7.0`
- `pandas>=2.2.0` for CSV export, acceptable to replace with Python `csv` if preferred
- `python-dateutil>=2.9.0`
- Do not add `torch`

Use a standard `build-system` with `hatchling` or `setuptools`. Expose a console script entry point `breadscribe = breadscribe.cli:app`.

## README content to generate
Include the following sections and examples.

**What it is**
- Breadscribe runs locally with open source models, transcribes video and audio to sidecar text, and can also generate subtitles.

**Install**
```

pipx install .

```
or
```

python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .

```

**Quick start**
```

breadscribe /path/to/folder
breadscribe /path/to/clip.mp4 --srt
breadscribe /clips --model large-v3 --device cuda --compute-type float16
breadscribe /clips --language en --initial-prompt "whole wheat, sourdough, autolyse"

```

**Performance tips**
- NVIDIA GPU: `--device cuda --compute-type float16` is recommended.
- CPU only: keep `--model distil-large-v2` or `medium`, default `int8` compute type.
- Apple Silicon: CPU `int8` works well for `small` to `distil-large-v2`. If you want Apple GPU acceleration, consider `whisper.cpp` as an optional alternative. Provide a short note and example command in the README, no integration is required.

**About VAD and stability**
- VAD reduces false text in silence.
- The defaults `temperature=0` and `condition_on_previous_text=False` help reduce run on hallucinations.
- Check `.segments.csv` for quick spot checks on important videos.

**Media decoding note**
- The tool relies on PyAV. If your system needs FFmpeg, include short install tips for macOS, Windows, and Linux.

**First run**
- The first run downloads model weights to a local cache. Subsequent runs reuse them.

## Tests
- `tests/test_subtitles.py` should validate SRT and VTT formatting for a small list of synthetic segments.
- `tests/test_cli.py` should test CLI argument parsing and output path derivation without running a full model. Use temporary files and mocks for the transcriber call.

## CI
Add `.github/workflows/ci.yml` with a simple GitHub Actions workflow:
- Run on pushes and pull requests.
- Set up Python 3.10 on Ubuntu.
- Install the project with test extras.
- Run the tests. Do not download large models, do not require CUDA.

## Coding style and UX
- Use `rich` for clear progress and error messages.
- Print detected language per file.
- Print a final summary with number of files processed and average real time factor.
- Fail gracefully for individual file errors.

## License
Include an MIT license file.

## Deliverables
- All source files listed above.
- A concise README with the sections described.
- A minimal CI workflow.
- No placeholders. Provide complete, runnable code.

## Now implement it
Generate the full project with the exact structure, all source files, tests, README, and CI workflow. Use the defaults and behaviors listed in this prompt. Ensure the CLI installs an entry point named `breadscribe`. Ensure the tool writes `basename.txt` next to each input by default, with optional `.srt`, `.vtt`, and `.segments.csv` when requested.
