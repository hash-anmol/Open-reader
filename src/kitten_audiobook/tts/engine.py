from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    from kittentts import KittenTTS  # type: ignore
except Exception as e:  # pragma: no cover
    KittenTTS = None  # type: ignore


@dataclass
class TTSConfig:
    model_id: str = "KittenML/kitten-tts-nano-0.1"
    default_voice: str = "expr-voice-4-f"
    default_speed: float = 1.0


class TTSEngine:
    def __init__(self, config: Optional[TTSConfig] = None) -> None:
        if config is None:
            config = TTSConfig()
        self.config = config
        if KittenTTS is None:
            raise RuntimeError("kittentts package not installed. Install wheel from KittenML releases.")
        self._model = KittenTTS(self.config.model_id)

    @property
    def available_voices(self) -> List[str]:
        try:
            return list(self._model.available_voices)
        except Exception:
            # Fallback common set
            return [
                'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',
                'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f'
            ]

    def synthesize(self, text: str, voice: Optional[str] = None, speed: Optional[float] = None) -> np.ndarray:
        v = voice or self.config.default_voice
        s = speed if speed is not None else self.config.default_speed
        audio = self._model.generate(text, voice=v, speed=s)
        return audio  # float32 mono 24kHz
