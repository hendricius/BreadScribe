from pathlib import Path
import tempfile

from breadscribe.io_utils import discover_inputs, ALLOWED_EXTS


def test_discover_skips_hidden_by_default():
	with tempfile.TemporaryDirectory() as d:
		root = Path(d)
		(root / "a.mov").write_bytes(b"")
		(root / ".hidden.mov").write_bytes(b"")
		inputs = discover_inputs(root, recursive=False, allowed_exts=ALLOWED_EXTS)
		assert (root / "a.mov") in inputs
		assert (root / ".hidden.mov") not in inputs


def test_discover_include_hidden_when_flag_set():
	with tempfile.TemporaryDirectory() as d:
		root = Path(d)
		(root / "a.mov").write_bytes(b"")
		(root / ".hidden.mov").write_bytes(b"")
		inputs = discover_inputs(root, recursive=False, allowed_exts=ALLOWED_EXTS, include_hidden=True)
		assert (root / "a.mov") in inputs
		assert (root / ".hidden.mov") in inputs


