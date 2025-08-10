from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Optional

import numpy as np
import pyloudnorm as pyln


def loudness_normalize(audio: np.ndarray, sample_rate: int = 24000, target_lufs: float = -18.0) -> np.ndarray:
    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(audio)
    return pyln.normalize.loudness(audio, loudness, target_lufs)


def encode_mp3(input_wav: Path, output_mp3: Path, bitrate: str = "80k") -> None:
    output_mp3.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_wav),
        "-vn",
        "-ac", "1",
        "-ar", "24000",
        "-b:a", bitrate,
        str(output_mp3),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
