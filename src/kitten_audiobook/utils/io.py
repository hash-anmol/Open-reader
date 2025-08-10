from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path


DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = Path("output") / "test_mp3"


def ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sha1_of_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def chunk_cache_path(text: str, voice: str, speed: float, model_version: str = "kitten-tts-nano-0.1") -> Path:
    key = f"{text}\n<voice>{voice}</voice>\n<speed>{speed}</speed>\n<model>{model_version}</model>"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"chunk_{digest}.wav"


@dataclass
class OutputTarget:
    directory: Path
    base_name: str

    @property
    def wav_path(self) -> Path:
        return self.directory / f"{self.base_name}.wav"

    @property
    def mp3_path(self) -> Path:
        return self.directory / f"{self.base_name}.mp3"


def derive_output_target(input_pdf: Path, out_dir: Path | None = None) -> OutputTarget:
    directory = out_dir if out_dir is not None else OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    base_name = input_pdf.stem
    return OutputTarget(directory=directory, base_name=base_name)
