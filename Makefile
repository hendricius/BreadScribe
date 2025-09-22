SHELL := /bin/bash

VENV := .venv
BREADSCRIBE := $(VENV)/bin/breadscribe
BREADSCRIBE_BIN := $(if $(wildcard $(BREADSCRIBE)),$(BREADSCRIBE),breadscribe)
PYTHON ?= python3

.PHONY: setup lint test test-sample transcribe transcribe-sample clean
setup:
	@if [ ! -x "$(BREADSCRIBE)" ]; then \
		echo "[setup] Creating venv and installing breadscribe..."; \
		$(PYTHON) -m venv $(VENV); \
		$(VENV)/bin/python -m pip install --upgrade pip >/dev/null; \
		$(VENV)/bin/pip install -e .[test,lint] >/dev/null; \
	fi
	@# Ensure lint tooling is present even if venv already existed
	@"$(VENV)/bin/ruff" --version >/dev/null 2>&1 || "$(VENV)/bin/pip" install -q "ruff>=0.6.3"

.PHONY: lint
lint: setup
	$(VENV)/bin/ruff check .

.PHONY: transcribe
transcribe: setup ## Usage: make transcribe FOLDER="/path/with spaces" [FLAGS="--srt ..."]
	@if [ -z "$(FOLDER)" ]; then \
		echo "Usage: make transcribe FOLDER=\"/path/with spaces\" [FLAGS=\"--srt --vtt --segments-csv --recursive\"]"; \
		exit 1; \
	fi; \
	echo "Transcribing: $(FOLDER)"; \
	FILE_PATH="$(FOLDER)"; \
	DEFAULT_FLAGS="--recursive --overwrite --language en --strict-no-speech --clean-fillers --srt --vtt --model tiny.en --print-text"; \
	"$(BREADSCRIBE_BIN)" "$$FILE_PATH" --overwrite $$DEFAULT_FLAGS $(FLAGS); \
	# Only do small-output fallback for single-file inputs \
	if [ -f "$$FILE_PATH" ]; then \
		TXT="$${FILE_PATH%.*}.txt"; \
		SZ=$$(wc -c < "$$TXT" 2>/dev/null || echo 0); \
		if [ "$$SZ" -le 80 ]; then \
			echo "[warn] small output ($$SZ bytes), retrying with --stream-index 0"; \
			"$(BREADSCRIBE_BIN)" "$$FILE_PATH" --overwrite $$DEFAULT_FLAGS --stream-index 0 $(FLAGS); \
			SZ=$$(wc -c < "$$TXT" 2>/dev/null || echo 0); \
		fi; \
		if [ "$$SZ" -le 80 ]; then \
			echo "[warn] small output ($$SZ bytes), retrying with --stream-index 1"; \
			"$(BREADSCRIBE_BIN)" "$$FILE_PATH" --overwrite $$DEFAULT_FLAGS --stream-index 1 $(FLAGS); \
			SZ=$$(wc -c < "$$TXT" 2>/dev/null || echo 0); \
		fi; \
		if [ "$$SZ" -le 80 ]; then \
			echo "[warn] Small transcript ($$SZ bytes) for $$FILE_PATH; continuing"; \
		else \
			echo "[ok] output size $$SZ bytes for $$TXT"; \
		fi; \
	fi


.PHONY: test
test: setup
	$(VENV)/bin/pytest -q -m "not slow"

.PHONY: test-sample
test-sample: setup
	$(VENV)/bin/pytest -q -m slow

.PHONY: transcribe-sample
transcribe-sample: setup ## Transcribe sample media with safe defaults
	"$(BREADSCRIBE_BIN)" sample --model tiny.en --workers 1 --clean-fillers --strict-no-speech --srt --vtt || true
	@echo "---- OUTPUTS ----"; for f in sample/*.txt; do echo $$f; sed -n '1,12p' $$f; echo ""; done

.PHONY: clean
clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build *.egg-info
	find . -name "*.pyc" -delete


