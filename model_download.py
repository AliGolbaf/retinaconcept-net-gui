#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional download of the public RetinaConcept-Net research checkpoints."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

from model_runtime import discover_checkpoints


CHECKPOINT_SOURCES = {
    "blackbox": ("best_model.pt", "1K8agioezWh8Tasy5fnfCtANZOHFZ3W7N"),
    "cbm": ("best_cbm_model.pt", "1pc2ZFBo6eNpxS8xTbJrhxcypERiUqPtC"),
}


def download_models(folder: Path, downloader: Callable | None = None) -> dict:
    """
    Download missing state-dictionary checkpoints from Google Drive.

    Existing checkpoints are preserved. Each download is staged in a
    temporary folder and moved into models/ only after it completes.

    Args:
        folder: Local directory for trained model checkpoints.
        downloader: Optional gdown-compatible callable for testing.

    Returns:
        Dictionary containing downloaded, skipped and failed model names.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    if downloader is None:
        try:
            import gdown
        except ImportError as error:
            raise RuntimeError("Install gdown using: pip install -r requirements.txt") from error
        downloader = gdown.download

    discovered, problems = discover_checkpoints(folder)
    result = {"downloaded": {}, "skipped": {}, "errors": {}}

    for kind, (filename, file_id) in CHECKPOINT_SOURCES.items():
        if kind in discovered:
            existing = discovered[kind]
            if existing.stat().st_size == 0:
                result["errors"][kind] = f"{existing.name} is empty; remove or replace it manually."
            else:
                result["skipped"][kind] = existing.name
            continue

        if problems.get(kind, "").startswith("Multiple checkpoints"):
            result["errors"][kind] = problems[kind]
            continue

        destination = folder / filename
        try:
            with TemporaryDirectory(prefix=".model-download-", dir=folder) as temporary:
                temporary_file = Path(temporary) / filename
                downloaded = downloader(id=file_id, output=str(temporary_file), quiet=True)
                if not downloaded or not temporary_file.is_file() or temporary_file.stat().st_size < 1024:
                    raise RuntimeError("Download was unsuccessful or returned an incomplete file.")
                if destination.exists():
                    raise RuntimeError(f"{filename} already exists; it will not be overwritten.")
                temporary_file.replace(destination)
            result["downloaded"][kind] = filename
        except Exception as error:
            result["errors"][kind] = str(error).splitlines()[0]

    return result


if __name__ == "__main__":
    downloaded = download_models(Path(__file__).resolve().parent / "models")
    for category in ("downloaded", "skipped", "errors"):
        for kind, message in downloaded[category].items():
            print(f"{kind} ({category}): {message}")
    if downloaded["errors"]:
        raise SystemExit(1)
