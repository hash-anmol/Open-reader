from __future__ import annotations

import gradio as gr
import numpy as np

from ..tts.engine import TTSEngine, TTSConfig
from ..tts.synthesis import synthesize_with_backoff

# Minimal Kokoro voices allowlist (subset). See VOICES.md for full list.
KOKORO_VOICES = {
    'af_heart', 'af_nova', 'af_alloy', 'af_aoede', 'af_bella', 'af_jessica', 'af_kore', 'af_nicole', 'af_river', 'af_sarah', 'af_sky',
    'am_liam', 'am_adam', 'am_michael', 'am_onyx', 'am_eric', 'am_echo', 'am_fenrir', 'am_puck',
    'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily',
    'bm_daniel', 'bm_george', 'bm_lewis', 'bm_fable',
    'ef_dora', 'em_alex', 'em_santa', 'ff_siwis'
}

# Cache Kokoro pipeline in-memory to avoid reloading
_KOKORO_PIPE = None

def _get_kokoro_pipeline():
    global _KOKORO_PIPE
    if _KOKORO_PIPE is None:
        from kokoro import KPipeline
        _KOKORO_PIPE = KPipeline(lang_code='a')
    return _KOKORO_PIPE


def tts_generate(text: str, model_choice: str, voice: str, speed: float):
    if not text.strip():
        return 24000, np.zeros(2400, dtype=np.float32)  # 0.1s silence

    if model_choice == 'KittenTTS':
        engine = TTSEngine(TTSConfig())
        v = (voice or "expr-voice-4-f").strip()
        try:
            audio = engine.synthesize(text, voice=v, speed=speed)
        except Exception:
            audio = synthesize_with_backoff(engine, text, v, speed)
        if audio is None or len(audio) == 0:
            audio = np.zeros(2400, dtype=np.float32)
        return 24000, audio.astype(np.float32)

    if model_choice == 'Kokoro-82M':
        try:
            pipe = _get_kokoro_pipeline()
            v = (voice or 'af_heart').strip()
            if v not in KOKORO_VOICES:
                v = 'af_heart'

            def run_kokoro(vname: str):
                # Accumulate all yielded segments
                try:
                    gen = pipe(text, voice=vname)
                except Exception:
                    return None
                parts = []
                try:
                    for _, _, a in gen:
                        if a is not None and len(a) > 0:
                            parts.append(a.astype(np.float32))
                except Exception:
                    return None
                if not parts:
                    return None
                if len(parts) == 1:
                    return parts[0]
                return np.concatenate(parts)

            audio = run_kokoro(v)
            if audio is None or len(audio) == 0:
                audio = run_kokoro('af_heart')
            if audio is None or len(audio) == 0:
                audio = np.zeros(2400, dtype=np.float32)
            return 24000, audio.astype(np.float32)
        except Exception:
            return 24000, np.zeros(2400, dtype=np.float32)

    return 24000, np.zeros(2400, dtype=np.float32)


def build_ui():
    with gr.Blocks(title="Audiobook TTS") as demo:
        gr.Markdown("# Audiobook TTS")
        with gr.Row():
            model_choice = gr.Radio(["KittenTTS", "Kokoro-82M"], value="KittenTTS", label="Model")
            voice = gr.Textbox(value="expr-voice-4-f", label="Voice (KittenTTS or Kokoro voice id)")
            speed = gr.Slider(minimum=0.5, maximum=2.0, step=0.05, value=1.0, label="Speed")
        text = gr.Textbox(lines=5, label="Text")
        btn = gr.Button("Synthesize")
        audio = gr.Audio(label="Audio", autoplay=True, type="numpy")

        btn.click(fn=tts_generate, inputs=[text, model_choice, voice, speed], outputs=audio)
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
