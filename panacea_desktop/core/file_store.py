from __future__ import annotations

import hashlib
import mimetypes
import shutil
from dataclasses import dataclass
from pathlib import Path

from .db import _app_data_dir  # reuse location logic


@dataclass(frozen=True)
class StoredFile:
    storage_path: str
    sha256: str
    size: int
    mime: str
    original_name: str


def _files_dir() -> Path:
    d = _app_data_dir() / "files"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _hash_file(path: Path, *, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def import_file(path: str, *, category: str = "attachments") -> StoredFile:
    """Copy file into the app data folder and return stable metadata.

    We do this because original user file paths may become unavailable later
    (moved/deleted), especially in Windows EXE deployments.
    """
    src = Path(path)
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(str(src))

    sha = _hash_file(src)
    size = int(src.stat().st_size)
    mime, _ = mimetypes.guess_type(src.name)
    mime = mime or "application/octet-stream"

    ext = src.suffix.lower()
    safe_cat = "".join(ch for ch in category if ch.isalnum() or ch in ("-", "_")).strip() or "files"
    dst_dir = _files_dir() / safe_cat
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{sha}{ext}"

    if not dst.exists():
        shutil.copy2(str(src), str(dst))

    return StoredFile(
        storage_path=str(dst),
        sha256=sha,
        size=size,
        mime=mime,
        original_name=src.name,
    )
