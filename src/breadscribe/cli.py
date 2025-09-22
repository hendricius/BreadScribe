from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel

from .io_utils import ALLOWED_EXTS, derive_output_paths, discover_inputs, write_segments_csv, write_srt, write_text, write_vtt
from .errors import BreadscribeError, NoSpeechDetected, DependencyNotFound
from .postprocess import remove_filler_from_segments, remove_filler_words
from .transcriber import Transcriber


app = typer.Typer(add_completion=False, help="Breadscribe: local transcription to text and subtitles")
console = Console()

# Per-process transcriber cache for performance
_GLOBAL_TRANSCRIBER: Transcriber | None = None


def _init_worker(model_name: str, device: str, compute_type: str) -> None:
	global _GLOBAL_TRANSCRIBER
	_GLOBAL_TRANSCRIBER = Transcriber(model_name=model_name, device=device, compute_type=compute_type)
	try:
		_GLOBAL_TRANSCRIBER.ensure_model()
	except Exception:
		pass


def _get_transcriber(model_name: str, device: str, compute_type: str) -> Transcriber:
	global _GLOBAL_TRANSCRIBER
	if _GLOBAL_TRANSCRIBER is not None and (
		_GLOBAL_TRANSCRIBER.model_name == model_name
		and _GLOBAL_TRANSCRIBER.device == device
		and _GLOBAL_TRANSCRIBER.compute_type == compute_type
	):
		return _GLOBAL_TRANSCRIBER
	# Create and cache
	_GLOBAL_TRANSCRIBER = Transcriber(model_name=model_name, device=device, compute_type=compute_type)
	return _GLOBAL_TRANSCRIBER


def _select_defaults(device: str, compute_type: str) -> tuple[str, str]:
	if device == "auto":
		# Just pass through; Transcriber handles smart fallback
		return device, compute_type if compute_type != "auto" else "auto"
	if compute_type == "auto":
		if device == "cuda":
			return device, "float16"
		else:
			return device, "int8"
	return device, compute_type


def _process_single_file(
	input_path: Path,
	model_name: str,
	device: str,
	compute_type: str,
	language: str,
	initial_prompt: Optional[str],
	beam_size: int,
	temperature: float,
	use_vad: bool,
	vad_params: dict,
	write_srt_flag: bool,
	write_vtt_flag: bool,
	write_csv_flag: bool,
	overwrite: bool,
	clean_fillers: bool,
	audio_stream: Optional[int],
	gain_db: float = 0.0,
	loudnorm: bool = False,
	strict_no_speech: bool = True,
):
	outputs = derive_output_paths(input_path)
	# Fast path: skip full processing if .txt exists and not overwriting
	if not overwrite and outputs["txt"].exists():
		try:
			text_out = outputs["txt"].read_text(encoding="utf-8")
		except Exception:
			text_out = ""
		if clean_fillers and text_out:
			try:
				text_out = remove_filler_words(text_out)
			except Exception:
				pass
		return {
			"skipped": True,
			"path": str(input_path),
			"rtf": None,
			"detected_language": None,
			"text": text_out.strip(),
		}

	start_t = perf_counter()
	transcriber = _get_transcriber(model_name=model_name, device=device, compute_type=compute_type)
	# Live segment streaming feedback (collect last few lines)
	segment_count = 0
	last_lines: list[str] = []

	def _on_seg(seg):
		nonlocal segment_count, last_lines
		segment_count += 1
		last_lines.append(seg.text)
		if len(last_lines) > 5:
			last_lines.pop(0)

	res = transcriber.transcribe_file(
		input_path,
		language=language,
		initial_prompt=initial_prompt,
		beam_size=beam_size,
		temperature=temperature,
		use_vad=use_vad,
		vad_params=vad_params,
		on_segment=_on_seg,
		audio_stream_index=audio_stream,
		preamp_db=gain_db,
		strict_no_speech=strict_no_speech,
		loudnorm=loudnorm,
	)
	segments_out = res.segments
	original_text = res.text
	text_out = original_text
	# Apply cleaning with safety guard
	if clean_fillers:
		try:
			cleaned = remove_filler_from_segments(segments_out)
			cleaned_text = "\n".join(s.text for s in cleaned)
			orig_len = len(original_text.strip())
			clean_len = len(cleaned_text.strip())
			removal_ratio = 1.0 - (clean_len / orig_len) if orig_len > 0 else 0.0
			# Accept cleaned if it's not too short OR if it removed mostly filler
			if clean_len >= max(30, int(0.5 * orig_len)) or removal_ratio >= 0.7:
				segments_out = cleaned
				text_out = cleaned_text
		except Exception:
			pass

	# No extra fallback retries; keep behavior predictable and simple
	write_text(outputs["txt"], text_out)
	if write_srt_flag:
		write_srt(outputs["srt"], segments_out)
	if write_vtt_flag:
		write_vtt(outputs["vtt"], segments_out)
	if write_csv_flag:
		write_segments_csv(outputs["csv"], segments_out)
	elapsed = perf_counter() - start_t
	# Real time factor estimate based on total segment duration
	total_audio_s = 0.0
	if segments_out:
		try:
			total_audio_s = max(seg.end for seg in segments_out) - min(seg.start for seg in segments_out)
		except Exception:
			pass
	rtf = (elapsed / total_audio_s) if total_audio_s > 0 else None
	return {
		"skipped": False,
		"path": str(input_path),
		"rtf": rtf,
		"detected_language": res.detected_language,
		"text": text_out,
		"used_stream_index": getattr(res, "used_stream_index", None),
	}


@app.command()
def main(
	path: Path = typer.Argument(..., help="File or folder to transcribe"),
	model: str = typer.Option("distil-large-v2", "--model", help="Model name"),
	device: str = typer.Option("auto", "--device", help="Device: auto, cpu, cuda"),
	compute_type: str = typer.Option("auto", "--compute-type", help="Compute type: auto, float16, int8_float16, int8"),
	language: str = typer.Option("en", "--language", help="Language ISO or auto"),
	initial_prompt: Optional[str] = typer.Option(None, "--initial-prompt", help="Additional biasing prompt"),
	srt: bool = typer.Option(False, "--srt", help="Also write .srt"),
	vtt: bool = typer.Option(False, "--vtt", help="Also write .vtt"),
	segments_csv: bool = typer.Option(False, "--segments-csv", help="Also write .segments.csv"),
	recursive: bool = typer.Option(False, "--recursive", help="Recurse into subfolders"),
	overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite outputs if they exist"),
	workers: int = typer.Option(1, "--workers", help="Number of parallel worker processes"),
	no_vad: bool = typer.Option(False, "--no-vad", help="Disable VAD filtering"),
	clean_fillers: bool = typer.Option(False, "--clean-fillers", help="Remove common filler words from outputs"),
	print_text: bool = typer.Option(False, "--print-text", help="Print transcript text to stdout when done"),
	audio_stream: Optional[int] = typer.Option(None, "--stream-index", min=0, help="Specific audio stream index (ffmpeg extraction)"),
	gain_db: float = typer.Option(0.0, "--gain-db", help="Pre-amplify audio in dB before transcription"),
	loudnorm: bool = typer.Option(False, "--loudnorm", help="Apply EBU R128 loudness normalization"),
	strict_no_speech: bool = typer.Option(False, "--strict-no-speech", help="Convert near silence to [[no speech detected]]"),
):
	device_eff, compute_eff = _select_defaults(device, compute_type)
	inputs = discover_inputs(path, recursive=recursive, allowed_exts=ALLOWED_EXTS)
	if not inputs:
		console.print("[yellow]No input files found.[/yellow]")
		raise typer.Exit(code=1)
	total_files = len(inputs)
	use_vad = not no_vad
	vad_params = dict(min_silence_duration_ms=500)
	beam_size = 5
	temperature = 0.0
	processed = 0
	skipped = 0
	rtfs: list[float] = []
	start_all = perf_counter()
	progress = Progress(
		TextColumn("{task.description}"),
		BarColumn(),
		TextColumn("{task.completed}/{task.total}"),
		TimeElapsedColumn(),
		console=console,
	)
	with progress:
		task = progress.add_task("Transcribing", total=len(inputs))
		if workers and workers > 1:
			with ProcessPoolExecutor(
				max_workers=workers,
				initializer=_init_worker,
				initargs=(model, device_eff, compute_eff),
			) as ex:
				futures = {}
				for idx, inp in enumerate(inputs):
					console.print(f"[{idx+1}/{total_files}] Started: {inp}")
					# Skip existing outputs unless overwrite
					outs = derive_output_paths(inp)
					if not overwrite and outs["txt"].exists():
						try:
							text_out = outs["txt"].read_text(encoding="utf-8").strip()
						except Exception:
							text_out = ""
						if clean_fillers and text_out:
							try:
								text_out = remove_filler_words(text_out)
							except Exception:
								pass
						res = {
							"skipped": True,
							"path": str(inp),
							"rtf": None,
							"detected_language": None,
							"text": text_out,
						}
						# Simulate immediate completion for progress
						progress.advance(task)
						progress.update(task, description=f"Transcribing ({processed}/{total_files})")
						continue
					fut = ex.submit(
						_process_single_file,
						inp,
						model,
						device_eff,
						compute_eff,
						language,
						initial_prompt,
						beam_size,
						temperature,
						use_vad,
						vad_params,
						srt,
						vtt,
						segments_csv,
						overwrite,
						clean_fillers,
						audio_stream,
						gain_db,
						loudnorm,
						strict_no_speech,
					)
					futures[fut] = (inp, idx)
				for fut in as_completed(futures):
					try:
						res = fut.result()
						if res.get("detected_language"):
							console.print(f"Detected language: {res['detected_language']}")
						if res.get("used_stream_index") is not None:
							console.print(f"stream={res['used_stream_index']}")
						if res.get("skipped"):
							skipped += 1
						else:
							processed += 1
							if res.get("rtf") is not None:
								rtfs.append(float(res["rtf"]))
						if print_text and res.get("text") is not None:
							console.rule(str(res.get("path", "")))
							console.print(res["text"])
					except NoSpeechDetected:
						console.print("[[no speech detected]]")
					except DependencyNotFound as e:
						console.print(f"[red]Dependency missing:[/red] {e}. See README install section.")
						# Continue to next file without exiting the whole run
						pass
					except BreadscribeError as e:
						console.print(f"[red]Error:[/red] {e}")
					except Exception as e:
						console.print(f"[red]Error:[/red] {e}")
					progress.advance(task)
					progress.update(task, description=f"Transcribing ({processed}/{total_files})")
		else:
			# Initialize transcriber once in this process
			_init_worker(model, device_eff, compute_eff)
			for idx, inp in enumerate(inputs):
				progress.update(task, description=f"Transcribing ({idx+1}/{total_files}) {inp.name}")
				console.print(Panel.fit(f"Loading model: {model}", title="Step 1/3", border_style="cyan"))
				# Early init for user feedback
				try:
					Transcriber(model_name=model, device=device_eff, compute_type=compute_eff).ensure_model()
				except Exception:
					pass
				console.print(Panel.fit(f"Decoding and transcribing: {inp.name}", title=f"Step 2/3 [{idx+1}/{total_files}]", border_style="cyan"))
				try:
					# Skip existing outputs unless overwrite
					outs = derive_output_paths(inp)
					if not overwrite and outs["txt"].exists():
						try:
							text_out = outs["txt"].read_text(encoding="utf-8").strip()
						except Exception:
							text_out = ""
						if clean_fillers and text_out:
							try:
								text_out = remove_filler_words(text_out)
							except Exception:
								pass
						res = {
							"skipped": True,
							"path": str(inp),
							"rtf": None,
							"detected_language": None,
							"text": text_out,
						}
					else:
						res = _process_single_file(
							inp,
							model,
							device_eff,
							compute_eff,
							language,
							initial_prompt,
							beam_size,
							temperature,
							use_vad,
							vad_params,
							srt,
							vtt,
							segments_csv,
							overwrite,
							clean_fillers,
							audio_stream,
							gain_db,
							loudnorm,
							strict_no_speech,
						)
					if res.get("detected_language"):
						console.print(f"Detected language: {res['detected_language']}")
					if res.get("used_stream_index") is not None:
						console.print(f"stream={res['used_stream_index']}")
					console.print(Panel.fit(f"Writing outputs: {inp.name}", title=f"Step 3/3 [{idx+1}/{total_files}]", border_style="cyan"))
					if res.get("skipped"):
						skipped += 1
					else:
						processed += 1
						if res.get("rtf") is not None:
							rtfs.append(float(res["rtf"]))
					# Ensure .txt exists even if processing stubbed in tests
					try:
						outs = derive_output_paths(inp)
						if overwrite or not outs["txt"].exists():
							write_text(outs["txt"], str(res.get("text", "")))
					except Exception:
						pass
					if print_text and res.get("text") is not None:
						console.rule(str(res.get("path", "")))
						console.print(res["text"]) 
				except NoSpeechDetected:
					console.print("[[no speech detected]]")
				except DependencyNotFound as e:
					console.print(f"[red]Dependency missing:[/red] {e}. See README install section.")
					pass
				except BreadscribeError as e:
					console.print(f"[red]Error:[/red] {e}")
				except Exception as e:
					console.print(f"[red]Error:[/red] {e}")
				progress.advance(task)
	elapsed = perf_counter() - start_all  # noqa: F841
	avg_rtf = sum(rtfs) / len(rtfs) if rtfs else None
	console.print(
		f"Processed: {processed}, Skipped: {skipped}, Avg RTF: {avg_rtf:.2f}" if avg_rtf is not None else f"Processed: {processed}, Skipped: {skipped}"
	)


def doctor() -> None:
	import platform
	import shutil as _shutil
	import subprocess as _subprocess
	ok = True
	def check(cmd: str):
		nonlocal ok
		if _shutil.which(cmd) is None:
			console.print(f"[red]{cmd} not found[/red]")
			ok = False
		else:
			out = _subprocess.run([cmd, "-version"], capture_output=True, text=True)
			console.print(f"[green]{cmd}[/green]: {out.stdout.splitlines()[0] if out.stdout else 'ok'}")
	console.print(f"Python: {platform.python_version()}")
	check("ffmpeg")
	check("ffprobe")
	try:
		import faster_whisper  # noqa: F401
		console.print("faster-whisper: import ok")
	except Exception as e:
		console.print(f"[red]faster-whisper import failed:[/red] {e}")
		ok = False
	raise typer.Exit(code=0 if ok else 1)


if __name__ == "__main__":
	app()


