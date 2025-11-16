"""
Robust stub (cache) read/write utilities.

Features:
- Path None-safe
- Directory auto-creation
- Atomic writes using a temporary file + os.replace
- Optional gzip compression (disabled by default)
- Error handling with informative returns (no unhandled exceptions)
"""

from __future__ import annotations
import os
import pickle
import tempfile
from typing import Any, Optional
import gzip
import errno


def _ensure_dir_for_file(path: str) -> None:
    """Create parent directory for path if it doesn't exist (race-safe)."""
    parent = os.path.dirname(path)
    if not parent:
        return
    try:
        os.makedirs(parent, exist_ok=True)
    except OSError as e:
        # If directory already exists (race), ignore; otherwise re-raise
        if e.errno != errno.EEXIST:
            raise


def save_stub(stub_path: Optional[str], obj: Any, compress: bool = False) -> bool:
    """
    Save a Python object to disk safely.

    Args:
        stub_path: target file path. If None, function is a no-op and returns False.
        obj: picklable Python object to save.
        compress: if True, save using gzip compression (.gz not enforced automatically).

    Returns:
        bool: True on success, False on failure (no exception raised).
    """
    if stub_path is None:
        return False

    try:
        _ensure_dir_for_file(stub_path)

        # Write to a temp file in the same dir for atomic replace
        dir_name = os.path.dirname(stub_path) or "."
        with tempfile.NamedTemporaryFile(mode="wb", dir=dir_name, delete=False) as tf:
            tmp_name = tf.name
            if compress:
                # Write compressed pickle
                with gzip.GzipFile(fileobj=tf, mode="wb") as gz:
                    pickle.dump(obj, gz)
            else:
                pickle.dump(obj, tf)

        # Atomic replace
        os.replace(tmp_name, stub_path)
        return True

    except Exception as e:
        # Don't raise — return False so callers can continue.
        # If you want logging, add logging here or print (kept quiet to not spam).
        # print(f"[save_stub] Failed to save stub {stub_path}: {e}")
        try:
            # try to clean temp file if exists
            if 'tmp_name' in locals() and os.path.exists(tmp_name):
                os.remove(tmp_name)
        except Exception:
            pass
        return False


def read_stub(read_from_stub: bool, stub_path: Optional[str], compress: bool = False) -> Optional[Any]:
    """
    Read a previously saved Python object from disk if available.

    Args:
        read_from_stub: whether to attempt reading from disk.
        stub_path: file path where the object should be. If None, returns None.
        compress: if True, read gzip-compressed pickle.

    Returns:
        object: the unpickled object if successful, otherwise None.
    """
    if not read_from_stub or stub_path is None:
        return None

    if not os.path.exists(stub_path):
        return None

    try:
        if compress:
            with gzip.open(stub_path, "rb") as f:
                return pickle.load(f)
        else:
            with open(stub_path, "rb") as f:
                return pickle.load(f)
    except Exception:
        # Failed to read/deserialize — return None
        # print(f"[read_stub] Failed to read stub {stub_path}: {e}")
        return None