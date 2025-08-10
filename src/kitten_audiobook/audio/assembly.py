from __future__ import annotations

from pathlib import Path
import numpy as np
import soundfile as sf


def save_wav(path: Path, audio: np.ndarray, sample_rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype('float32'), sample_rate)
