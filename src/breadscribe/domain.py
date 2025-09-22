from __future__ import annotations

from functools import lru_cache
from importlib import resources as importlib_resources
import os
from pathlib import Path
from typing import Optional


_DEFAULT_GLOSSARY: list[str] = [
	"autolyse",
	"levain",
	"poolish",
	"preferment",
	"bulk fermentation",
	"crumb",
	"oven spring",
	"banneton",
	"hydration",
	"baker’s percent",
	"lamination",
	"stretch and fold",
	"slap and fold",
	"scoring",
	"couche",
	"pâte fermentée",
	"tangzhong",
	"yudane",
	"retard",
	"aliquot jar",
	"levain build",
	"shaping",
	"batard",
	"boule",
	"preheat",
	"steam",
	"Dutch oven",
	"crumb shot",
	"to 96 F internal",
]


def _read_glossary_file(path: Path) -> list[str]:
	try:
		lines = path.read_text(encoding="utf-8").splitlines()
		terms: list[str] = []
		for line in lines:
			line = line.strip()
			if not line or line.startswith("#"):
				continue
			terms.append(line)
		# Deduplicate preserving order
		seen = set()
		unique_terms: list[str] = []
		for t in terms:
			if t not in seen:
				seen.add(t)
				unique_terms.append(t)
		return unique_terms
	except Exception:
		return []


@lru_cache(maxsize=1)
def load_glossary() -> list[str]:
	# Priority: env var path > CWD glossary.txt > packaged glossary.txt > defaults
	env_path = os.environ.get("BREADSCRIBE_GLOSSARY")
	if env_path:
		terms = _read_glossary_file(Path(env_path))
		if terms:
			return terms
	local = Path.cwd() / "glossary.txt"
	terms = _read_glossary_file(local)
	if terms:
		return terms
	try:
		res = importlib_resources.files("breadscribe").joinpath("glossary.txt")
		if res.is_file():
			terms = _read_glossary_file(Path(res))
			if terms:
				return terms
	except Exception:
		pass
	return _DEFAULT_GLOSSARY


BAKING_GLOSSARY: list[str] = load_glossary()


def build_initial_prompt(user_prompt: Optional[str]) -> str:
	terms = ", ".join(BAKING_GLOSSARY)
	base = f"This video may contain baking terms such as {terms}."
	if user_prompt and user_prompt.strip():
		return base + " " + user_prompt.strip()
	return base



