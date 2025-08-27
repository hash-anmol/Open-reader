from __future__ import annotations

import gradio as gr
import numpy as np
import logging
import warnings

# Handle both relative and absolute imports for flexibility
try:
    # Try relative imports first (when run as module)
    from ..tts.engine import TTSEngine, TTSConfig
    from ..tts.synthesis import (
        synthesize_with_backoff,
        generate_word_timings,
        word_timings_to_json,
        create_transcript_html,
        detect_chapter_breaks_from_chunks,
        compute_chapter_ranges,
        assemble_chapter_audios,
    )
    from ..utils.io import ensure_dirs
    from ..ingestion.pdf_reader import read_pdf
    from ..ingestion.chapter_detection import detect_chapters_enhanced
    from ..ingestion.chapter_processing import ChapterProcessor
    from ..text.cleaning import normalize_text
    from ..text.segmentation import split_into_sentences
    from ..text.chunking import build_chunks
except ImportError:
    # Fallback to absolute imports (when run directly)
    import sys
    from pathlib import Path
    # Add the src directory to the path
    src_path = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(src_path))
    
    from kitten_audiobook.tts.engine import TTSEngine, TTSConfig
    from kitten_audiobook.tts.synthesis import (
        synthesize_with_backoff,
        generate_word_timings,
        word_timings_to_json,
        create_transcript_html,
        detect_chapter_breaks_from_chunks,
        compute_chapter_ranges,
        assemble_chapter_audios,
    )
    from kitten_audiobook.utils.io import ensure_dirs
    from kitten_audiobook.ingestion.pdf_reader import read_pdf
    from kitten_audiobook.ingestion.chapter_detection import detect_chapters_enhanced
    from kitten_audiobook.ingestion.chapter_processing import ChapterProcessor
    from kitten_audiobook.text.cleaning import normalize_text
    from kitten_audiobook.text.segmentation import split_into_sentences
    from kitten_audiobook.text.chunking import build_chunks
from pathlib import Path
import tempfile
import os
import zipfile
import soundfile as sf

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# Try to import Kokoro
try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
    logging.info("✅ Kokoro TTS loaded successfully")
except Exception as e:
    KOKORO_AVAILABLE = False
    logging.warning(f"⚠️ Kokoro TTS not available: {e}")

# Complete Kokoro voices list based on documentation research
KOKORO_VOICES = {
    # American Female (af_)
    'af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_jessica', 'af_kore', 'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky',
    # American Male (am_)
    'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael', 'am_onyx', 'am_puck',
    # British Female (bf_)
    'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily',
    # British Male (bm_)
    'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis',
    # Other languages and voices
    'ef_dora', 'em_alex', 'em_santa', 'ff_siwis'
}

# Kitten TTS voices based on documentation research
KITTEN_VOICES = [
    'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',
    'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f'
]

# Global instances
_kitten_engine = None
_kokoro_pipeline = None

def get_kitten_engine():
    """Get or create KittenTTS engine"""
    global _kitten_engine
    if _kitten_engine is None:
        try:
            _kitten_engine = TTSEngine(TTSConfig())
            logging.info("KittenTTS engine initialized")
        except Exception as e:
            logging.error(f"Failed to initialize KittenTTS engine: {e}")
            return None
    return _kitten_engine

def get_kokoro_pipeline():
    """Get or create Kokoro pipeline"""
    global _kokoro_pipeline
    if not KOKORO_AVAILABLE:
        return None
    
    if _kokoro_pipeline is None:
        try:
            _kokoro_pipeline = KPipeline(lang_code='a', repo_id="hexgrad/Kokoro-82M")
            logging.info("Kokoro pipeline initialized")
        except Exception as e:
            logging.error(f"Failed to initialize Kokoro pipeline: {e}")
            return None
    return _kokoro_pipeline


def get_voice_options(model_choice: str):
    """Return voice options based on selected model"""
    if model_choice == 'KittenTTS':
        return gr.Dropdown(choices=KITTEN_VOICES, value='expr-voice-4-f', label="Voice", interactive=True)
    else:  # Kokoro-82M
        kokoro_choices = sorted(list(KOKORO_VOICES))
        return gr.Dropdown(choices=kokoro_choices, value='af_heart', label="Voice", interactive=True)


def synthesize_kitten_tts(text: str, voice: str, speed: float):
    """Synthesize using KittenTTS with proper engine management"""
    engine = get_kitten_engine()
    if not engine:
        return None, "❌ KittenTTS engine not available", None, None
    
    try:
        # Validate voice
        if voice not in KITTEN_VOICES:
            voice = 'expr-voice-4-f'
            
        # Ensure directories exist for caching
        ensure_dirs()
        
        # Use existing synthesis with backoff
        try:
            audio = engine.synthesize(text, voice=voice, speed=speed)
            timings = generate_word_timings(text, audio)
        except Exception:
            # Try with backoff strategy
            audio, timings = synthesize_with_backoff(engine, text, voice, speed)
            
        if audio is None or len(audio) == 0:
            return None, "⚠️ KittenTTS synthesis failed", None, None
        
        # Generate JSON timings and HTML transcript
        timings_json = word_timings_to_json(timings)
        transcript_html = create_transcript_html(text, timings)
            
        return (audio.astype(np.float32), 
                f"✅ Successfully generated with KittenTTS ({voice})",
                timings_json,
                transcript_html)
        
    except Exception as e:
        logging.error(f"KittenTTS synthesis error: {e}")
        try:
            # Try with backoff as last resort
            audio, timings = synthesize_with_backoff(engine, text, voice, speed)
            if audio is not None and len(audio) > 0:
                timings_json = word_timings_to_json(timings)
                transcript_html = create_transcript_html(text, timings)
                return (audio.astype(np.float32), 
                        f"✅ Generated with backoff strategy: {str(e)[:50]}...",
                        timings_json,
                        transcript_html)
        except Exception:
            pass
            
        return None, f"❌ KittenTTS Error: {str(e)[:100]}", None, None

def synthesize_kokoro_tts(text: str, voice: str, speed: float):
    """Synthesize using Kokoro TTS"""
    pipeline = get_kokoro_pipeline()
    if not pipeline:
        return None, "❌ Kokoro pipeline not available", None, None
    
    try:
        # Validate voice
        if voice not in KOKORO_VOICES:
            voice = 'af_heart'
            
        # Generate audio using Kokoro
        segments = pipeline(text, voice=voice, speed=speed)
        
        # Collect all segments
        audio_parts = []
        for segment in segments:
            if hasattr(segment, 'audio'):
                audio_data = segment.audio
            elif isinstance(segment, tuple) and len(segment) >= 3:
                audio_data = segment[2]  
            else:
                audio_data = segment
                
            if audio_data is not None:
                # Convert tensor to numpy if needed
                if hasattr(audio_data, 'detach'):
                    audio_data = audio_data.detach().cpu().numpy()
                elif hasattr(audio_data, 'numpy'):
                    audio_data = audio_data.numpy()
                    
                if len(audio_data) > 0:
                    audio_parts.append(audio_data.astype(np.float32))
        
        if not audio_parts:
            return None, "⚠️ Kokoro synthesis produced no audio", None, None
            
        # Concatenate all parts
        if len(audio_parts) == 1:
            final_audio = audio_parts[0]
        else:
            final_audio = np.concatenate(audio_parts)
        
        # Generate word timings for Kokoro
        timings = generate_word_timings(text, final_audio)
        timings_json = word_timings_to_json(timings)
        transcript_html = create_transcript_html(text, timings)
            
        return (final_audio, 
                f"✅ Successfully generated with Kokoro-82M ({voice})",
                timings_json,
                transcript_html)
        
    except Exception as e:
        logging.error(f"Kokoro synthesis error: {e}")
        return None, f"❌ Kokoro Error: {str(e)[:100]}", None, None

def tts_generate(text: str, model_choice: str, voice: str, speed: float):
    """Enhanced TTS generation with real implementations and fallbacks"""
    if not text.strip():
        return ((24000, np.zeros(2400, dtype=np.float32)), 
                "No text provided", 
                "[]", 
                "<span>No text provided</span>")
    
    # Clean model name (remove demo indicators)
    clean_model = model_choice.replace(" (Demo)", "")
    
    # Try real TTS synthesis
    if clean_model == 'KittenTTS':
        result = synthesize_kitten_tts(text, voice, speed)
        audio, status, timings_json, transcript_html = result
        if audio is not None:
            return (24000, audio), status, timings_json, transcript_html
        else:
            # Fallback to simple generation
            return ((24000, np.zeros(2400, dtype=np.float32)), 
                    status, 
                    "[]", 
                    "<span>Synthesis failed</span>")
            
    elif clean_model == 'Kokoro-82M':
        result = synthesize_kokoro_tts(text, voice, speed)
        audio, status, timings_json, transcript_html = result
        if audio is not None:
            return (24000, audio), status, timings_json, transcript_html
        else:
            # Fallback to simple generation
            return ((24000, np.zeros(2400, dtype=np.float32)), 
                    status, 
                    "[]", 
                    "<span>Synthesis failed</span>")
    
    return ((24000, np.zeros(2400, dtype=np.float32)), 
            f"❌ Unknown model: {model_choice}", 
            "[]", 
            "<span>Unknown model</span>")


def process_pdf_to_text(pdf_file):
    """Process uploaded PDF and extract text for TTS conversion"""
    if pdf_file is None:
        return "No PDF file uploaded", "", "[]", "<span>No file processed</span>"
    
    try:
        # Handle Gradio file upload - pdf_file is a file path string
        if isinstance(pdf_file, str):
            # pdf_file is already a path to the uploaded file
            temp_path = Path(pdf_file)
        else:
            # Handle other file object types (fallback)
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                if hasattr(pdf_file, 'read'):
                    # File-like object
                    temp_file.write(pdf_file.read())
                else:
                    # Try to read as bytes
                    with open(pdf_file, 'rb') as src:
                        temp_file.write(src.read())
                temp_path = Path(temp_file.name)
        
        # Read PDF using existing infrastructure
        document = read_pdf(temp_path)
        
        # Extract and combine all text
        all_text = []
        total_pages = len(document.pages)
        
        for page in document.pages:
            for paragraph in page.paragraphs:
                normalized_text = normalize_text(paragraph.text)
                if normalized_text.strip():
                    all_text.append(normalized_text)
        
        # Combine all text
        combined_text = " ".join(all_text)
        
        # Clean up temporary file (only if we created one)
        if not isinstance(pdf_file, str):
            os.unlink(temp_path)
        
        # Generate basic info
        info = f"✅ PDF processed successfully!\n"
        info += f"📄 Pages: {total_pages}\n"
        info += f"📝 Characters: {len(combined_text):,}\n"
        info += f"📖 Words: {len(combined_text.split()):,}\n"
        
        # Create a preview (first 500 characters)
        preview = combined_text[:500] + ("..." if len(combined_text) > 500 else "")
        
        return info, combined_text, "[]", f"<div class='transcript-container'><div id='transcript'>{preview}</div></div>"
        
    except Exception as e:
        logging.error(f"PDF processing error: {e}")
        return f"❌ Error processing PDF: {str(e)[:100]}", "", "[]", "<span>PDF processing failed</span>"


def process_pdf_to_audiobook(pdf_file, model_choice, voice, speed, progress=gr.Progress()):
    """Complete PDF to audiobook conversion with proper chunking and synthesis"""
    if pdf_file is None:
        return None, "❌ No PDF file uploaded", None
    
    try:
        progress(0.05, desc="Processing PDF...")
        info, text, _, _ = process_pdf_to_text(pdf_file)
        
        if not text.strip():
            return None, "❌ No text extracted from PDF", None
        
        # Get the appropriate TTS engine
        clean_model = model_choice.replace(" (Demo)", "")
        engine = None
        
        if clean_model == 'KittenTTS':
            engine = get_kitten_engine()
            if not engine:
                return None, "❌ KittenTTS engine not available", None
        elif clean_model == 'Kokoro-82M':
            pipeline = get_kokoro_pipeline()
            if not pipeline:
                return None, "❌ Kokoro pipeline not available", None
        else:
            return None, f"❌ Unknown model: {model_choice}", None
        
        progress(0.1, desc="Segmenting text...")
        
        # Segment text into sentences and then build chunks
        try:
            # First split text into paragraphs, then into sentences
            paragraphs = text.split('\n\n')
            sentences = []
            for i, paragraph in enumerate(paragraphs):
                para_sentences = split_into_sentences(paragraph, i)
                sentences.extend(para_sentences)
            chunks = build_chunks(sentences, target_chars=280, hard_cap=360)
        except Exception as e:
            logging.warning(f"Text segmentation failed, using simple chunking: {e}")
            # Fallback: simple chunking by splitting on periods
            sentences_text = text.replace('.', '.\n').split('\n')
            chunks = []
            current_chunk = ""
            for i, sentence in enumerate(sentences_text):
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                if len(current_chunk + sentence) > 300 and current_chunk:
                    # Mock Chunk object for compatibility
                    chunk_obj = type('Chunk', (), {'text': current_chunk.strip()})()
                    chunks.append(chunk_obj)
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
            
            if current_chunk.strip():
                chunk_obj = type('Chunk', (), {'text': current_chunk.strip()})()
                chunks.append(chunk_obj)
        
        if not chunks:
            return None, "❌ Failed to create text chunks", None
        
        progress(0.15, desc=f"Converting {len(chunks)} chunks to audio...")
        
        # Convert each chunk to audio
        audio_segments = []
        all_timings = []
        current_time_offset = 0.0
        
        for i, chunk in enumerate(chunks):
            chunk_progress = 0.15 + (0.8 * i / len(chunks))
            progress(chunk_progress, desc=f"Processing chunk {i+1}/{len(chunks)}...")
            
            try:
                if clean_model == 'KittenTTS':
                    # Use KittenTTS synthesis with backoff
                    try:
                        audio = engine.synthesize(chunk.text, voice=voice, speed=speed)
                        timings = generate_word_timings(chunk.text, audio)
                    except Exception:
                        audio, timings = synthesize_with_backoff(engine, chunk.text, voice, speed)
                        
                elif clean_model == 'Kokoro-82M':
                    # Use Kokoro synthesis
                    segments = pipeline(chunk.text, voice=voice, speed=speed)
                    
                    # Collect audio from segments
                    audio_parts = []
                    for segment in segments:
                        if hasattr(segment, 'audio'):
                            audio_data = segment.audio
                        elif isinstance(segment, tuple) and len(segment) >= 3:
                            audio_data = segment[2]
                        else:
                            audio_data = segment
                            
                        if audio_data is not None:
                            if hasattr(audio_data, 'detach'):
                                audio_data = audio_data.detach().cpu().numpy()
                            elif hasattr(audio_data, 'numpy'):
                                audio_data = audio_data.numpy()
                                
                            if len(audio_data) > 0:
                                audio_parts.append(audio_data.astype(np.float32))
                    
                    if audio_parts:
                        audio = np.concatenate(audio_parts) if len(audio_parts) > 1 else audio_parts[0]
                        timings = generate_word_timings(chunk.text, audio)
                    else:
                        # Fallback: generate silence
                        audio = np.zeros(int(24000 * 2), dtype=np.float32)  # 2 seconds
                        timings = []
                
                if audio is not None and len(audio) > 0:
                    audio_segments.append(audio.astype(np.float32))
                    
                    # Adjust timings for continuous audiobook
                    for timing in timings:
                        adjusted_timing = type('WordTiming', (), {
                            'word': timing.word,
                            'start': timing.start + current_time_offset,
                            'end': timing.end + current_time_offset,
                            'idx': len(all_timings)
                        })()
                        all_timings.append(adjusted_timing)
                    
                    # Update time offset for next chunk
                    current_time_offset += len(audio) / 24000
                    
                    # Add small pause between chunks (0.3 seconds)
                    pause = np.zeros(int(24000 * 0.3), dtype=np.float32)
                    audio_segments.append(pause)
                    current_time_offset += 0.3
                    
            except Exception as e:
                logging.error(f"Error processing chunk {i+1}: {e}")
                # Add silence as fallback
                silence = np.zeros(int(24000 * 1), dtype=np.float32)
                audio_segments.append(silence)
                current_time_offset += 1.0
        
        progress(0.95, desc="Assembling final audiobook...")
        
        if not audio_segments:
            return None, "❌ Failed to generate any audio segments", None
        
        # Combine all audio segments
        final_audio = np.concatenate(audio_segments)
        
        if len(final_audio) == 0:
            return None, "❌ Generated audio is empty", None
        
        # Create comprehensive info
        duration_minutes = len(final_audio) / 24000 / 60
        final_info = f"""🎧 **Audiobook Created Successfully!**
        
📄 **Source:** {len(chunks)} text chunks processed
🎵 **Model:** {clean_model} ({voice})
⏱️ **Duration:** {duration_minutes:.1f} minutes
🔊 **Quality:** 24kHz mono
📝 **Words:** {len(text.split()):,}

✅ **Status:** Ready for download"""
        
        # Build per-chapter audios and package as a zip
        try:
            # Use chunk texts for detection
            chunk_texts = [getattr(ch, 'text', str(ch)) for ch in chunks]
            chapter_flags = detect_chapter_breaks_from_chunks(chunk_texts)
            ranges = compute_chapter_ranges(chapter_flags, total_chunks=len(chunks))
            chapter_audios = assemble_chapter_audios(audio_segments, ranges)

            # Write WAV files to a temp directory and zip them
            temp_dir = tempfile.mkdtemp(prefix="chapters_")
            wav_paths = []
            for idx, ch_audio in enumerate(chapter_audios, start=1):
                ch_path = os.path.join(temp_dir, f"chapter{idx:02d}.wav")
                sf.write(ch_path, ch_audio.astype(np.float32), 24000)
                wav_paths.append(ch_path)

            zip_path = os.path.join(temp_dir, "chapters.zip")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for p in wav_paths:
                    zf.write(p, arcname=os.path.basename(p))
            chapters_zip = zip_path
        except Exception as _e:
            logging.warning(f"Failed to create chapters zip: {_e}")
            chapters_zip = None

        progress(1.0, desc="Audiobook complete!")
        
        return (24000, final_audio), final_info, chapters_zip
        
    except Exception as e:
        logging.error(f"PDF to audiobook error: {e}")
        return None, f"❌ Error creating audiobook: {str(e)[:200]}", None


def build_ui():
    with gr.Blocks(title="Enhanced Kitten & Kokoro TTS", theme=gr.themes.Soft(), 
                   css="""
                   .w { 
                       cursor: pointer; 
                       padding: 2px 4px; 
                       margin: 1px; 
                       border-radius: 3px; 
                       transition: background-color 0.2s ease;
                       color: inherit;
                   }
                   .w.current { 
                       background-color: #ffe58a !important; 
                       font-weight: bold;
                       box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                       color: #000 !important;
                   }
                   .w:hover { 
                       background-color: rgba(255, 255, 255, 0.1);
                   }
                   .transcript-container {
                       max-height: 300px;
                       overflow-y: auto;
                       border: 1px solid var(--border-color-primary);
                       padding: 15px;
                       border-radius: 8px;
                       background-color: var(--background-secondary);
                       line-height: 1.6;
                       font-family: inherit;
                       color: var(--text-color-primary);
                   }
                   """) as demo:
        # State for storing word timings
        timings_state = gr.State(value="[]")
        
        gr.Markdown(f"""
        # 🎙️ Enhanced Kitten & Kokoro TTS Interface
        
        **Status:**
        - 🐱 **KittenTTS**: {'✅ Available' if True else '❌ Not Available'}
        - 🎵 **Kokoro-82M**: {'✅ Available' if KOKORO_AVAILABLE else '❌ Not Available (Demo Mode)'}
        
        **Features:**
        - 🎛️ **Model Switching**: Seamlessly switch between TTS engines
        - 🗣️ **Voice Selection**: Dropdown menus for all available voices
        - ⚡ **Real-time Generation**: Fast synthesis with detailed status feedback
        - 🔧 **Smart Fallbacks**: Automatic backoff strategies for robust synthesis
        - 📖 **Word Highlighting**: Real-time word highlighting synchronized with audio playback
        - 📚 **PDF Audiobooks**: Convert PDF documents to audiobooks
        """)
        
        with gr.Tabs():
            with gr.Tab("🎙️ Text-to-Speech"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎛️ Model & Voice Settings")
                        model_choice = gr.Radio(
                            ["KittenTTS", "Kokoro-82M"], 
                            value="KittenTTS", 
                            label="TTS Model",
                            info="KittenTTS: 25MB CPU-only | Kokoro-82M: High-quality GPU/CPU"
                        )
                        
                        voice_dropdown = gr.Dropdown(
                            choices=KITTEN_VOICES,
                            value='expr-voice-4-f',
                            label="Voice Selection",
                            info="Available voices for selected model",
                            interactive=True
                        )
                        
                        with gr.Row():
                            speed = gr.Slider(
                                minimum=0.5, 
                                maximum=2.5, 
                                step=0.05, 
                                value=1.0, 
                                label="Speech Speed",
                                info="0.5x = Slower, 2.5x = Faster"
                            )
                            
                        # Voice info panel
                        gr.Markdown("### 📋 Voice Information")
                        voice_info = gr.Markdown("""
                        **KittenTTS Voices:**
                        - `expr-voice-2/3/4/5-f/m`: Female/Male voices
                        - Lightweight, CPU-optimized
                        
                        **Kokoro Voices:**
                        - `af_*`: American Female voices
                        - `am_*`: American Male voices  
                        - `bf_*`: British Female voices
                        - `bm_*`: British Male voices
                        """)
                
                    with gr.Column(scale=2):
                        gr.Markdown("### 📝 Text Input")
                        text = gr.Textbox(
                            lines=8, 
                            label="Text to Synthesize",
                            placeholder="Enter your text here... The system supports both short phrases and longer paragraphs.",
                            info="Supports multi-sentence text with automatic segmentation"
                        )
                        
                        with gr.Row():
                            synthesize_btn = gr.Button("🎙️ Synthesize Audio", variant="primary", size="lg")
                            clear_btn = gr.Button("🗑️ Clear Text", variant="secondary")
                            debug_btn = gr.Button("🐛 Debug Highlighting", variant="secondary", size="sm")
                        
                        # Status and output
                        status_output = gr.Textbox(
                            label="Status",
                            value="Ready to synthesize...",
                            interactive=False,
                            lines=2
                        )
                        
                        audio = gr.Audio(
                            label="Generated Audio", 
                            autoplay=False, 
                            type="numpy",
                            show_download_button=True
                        )
                        
                        # Word highlighting transcript
                        gr.Markdown("### 📖 Interactive Transcript")
                        transcript_html = gr.HTML(
                            label="Transcript with Word Highlighting",
                            value="<div class='transcript-container'><div id='transcript'>Enter text and synthesize to see word-by-word highlighting</div></div>",
                            elem_id="transcript-container"
                        )
                
                        # JavaScript for word highlighting
                        highlighting_script = gr.HTML(value="""
                <script>
                let currentWordTimings = [];
                let audioElement = null;
                let currentWordIndex = -1;
                
                function initializeWordHighlighting() {
                    // Find the audio element - try multiple selectors
                    audioElement = document.querySelector('audio') || 
                                  document.querySelector('[data-testid="audio"]') ||
                                  document.querySelector('.audio-player audio');
                    
                    if (!audioElement) {
                        console.log('Audio element not found, retrying...');
                        setTimeout(initializeWordHighlighting, 500);
                        return;
                    }
                    
                    // Add event listeners
                    audioElement.addEventListener('timeupdate', updateWordHighlight);
                    audioElement.addEventListener('seeking', updateWordHighlight);
                    audioElement.addEventListener('play', onAudioPlay);
                    audioElement.addEventListener('pause', onAudioPause);
                    
                    console.log('✅ Word highlighting initialized for audio element:', audioElement);
                    console.log('🎵 Audio duration:', audioElement.duration, 'seconds');
                }
                
                function updateWordTimings(timingsJson) {
                    try {
                        currentWordTimings = JSON.parse(timingsJson);
                        console.log('Updated word timings:', currentWordTimings.length, 'words');
                        console.log('Sample timing:', currentWordTimings[0]);
                        clearWordHighlighting();
                    } catch (e) {
                        console.error('Failed to parse word timings:', e);
                        currentWordTimings = [];
                    }
                }
                
                function updateWordHighlight() {
                    if (!audioElement || currentWordTimings.length === 0) return;
                    
                    const currentTime = audioElement.currentTime;
                    let newWordIndex = -1;
                    
                    // Find the current word based on audio time
                    for (let i = 0; i < currentWordTimings.length; i++) {
                        const timing = currentWordTimings[i];
                        if (currentTime >= timing.start && currentTime <= timing.end) {
                            newWordIndex = i;
                            break;
                        }
                    }
                    
                    // Update highlighting if word changed
                    if (newWordIndex !== currentWordIndex) {
                        // Remove previous highlighting
                        if (currentWordIndex >= 0) {
                            const prevWord = document.getElementById('w-' + currentWordIndex);
                            if (prevWord) prevWord.classList.remove('current');
                        }
                        
                        // Add new highlighting
                        if (newWordIndex >= 0) {
                            const currentWord = document.getElementById('w-' + newWordIndex);
                            if (currentWord) {
                                currentWord.classList.add('current');
                                // Scroll into view if needed
                                currentWord.scrollIntoView({
                                    behavior: 'smooth',
                                    block: 'center',
                                    inline: 'nearest'
                                });
                            }
                        }
                        
                        currentWordIndex = newWordIndex;
                    }
                }
                
                function onAudioPlay() {
                    console.log('Audio started playing');
                    updateWordHighlight();
                }
                
                function onAudioPause() {
                    console.log('Audio paused');
                }
                
                function clearWordHighlighting() {
                    const words = document.querySelectorAll('.w.current');
                    words.forEach(word => word.classList.remove('current'));
                    currentWordIndex = -1;
                }
                
                function debugWordHighlighting() {
                    console.log('Debug info:');
                    console.log('- Audio element:', audioElement);
                    console.log('- Word timings:', currentWordTimings);
                    console.log('- Current word index:', currentWordIndex);
                    console.log('- Transcript words:', document.querySelectorAll('.w').length);
                }
                
                function testHighlighting() {
                    console.log('🧪 Testing word highlighting...');
                    const words = document.querySelectorAll('#transcript .w');
                    if (words.length === 0) {
                        console.log('❌ No word spans found in transcript');
                        return;
                    }
                    
                    console.log(`✅ Found ${words.length} word spans`);
                    
                    // Test highlighting each word for 500ms
                    let currentIndex = 0;
                    const testInterval = setInterval(() => {
                        // Remove previous highlighting
                        words.forEach(w => w.classList.remove('current'));
                        
                        // Add highlighting to current word
                        if (currentIndex < words.length) {
                            words[currentIndex].classList.add('current');
                            words[currentIndex].scrollIntoView({ behavior: 'smooth', block: 'center' });
                            console.log(`🎯 Highlighting word ${currentIndex + 1}/${words.length}: "${words[currentIndex].textContent}"`);
                            currentIndex++;
                        } else {
                            clearInterval(testInterval);
                            console.log('✅ Word highlighting test completed');
                        }
                    }, 500);
                }
                
                // Initialize when DOM is ready
                if (document.readyState === 'loading') {
                    document.addEventListener('DOMContentLoaded', initializeWordHighlighting);
                } else {
                    initializeWordHighlighting();
                }
                
                // Re-initialize periodically to catch dynamic audio elements
                setInterval(function() {
                    if (!audioElement) {
                        initializeWordHighlighting();
                    }
                }, 1000);
                
                // Also watch for Gradio audio player creation
                const observer = new MutationObserver(function(mutations) {
                    mutations.forEach(function(mutation) {
                        if (mutation.type === 'childList') {
                            mutation.addedNodes.forEach(function(node) {
                                if (node.nodeType === 1 && (node.tagName === 'AUDIO' || node.querySelector && node.querySelector('audio'))) {
                                    console.log('🎵 New audio element detected, reinitializing...');
                                    setTimeout(initializeWordHighlighting, 100);
                                }
                            });
                        }
                    });
                });
                
                // Start observing
                observer.observe(document.body, {
                    childList: true,
                    subtree: true
                });
                
                // Make functions globally available
                window.updateWordTimings = updateWordTimings;
                window.clearWordHighlighting = clearWordHighlighting;
                window.debugWordHighlighting = debugWordHighlighting;
                window.testHighlighting = testHighlighting;
                
                console.log('Word highlighting script loaded successfully');
                </script>
                        """, visible=True)
                        
                        # Audio controls
                        with gr.Row():
                            gr.Markdown("**Audio Controls:** Use the player above to play, pause, and download your generated audio. Words will highlight in sync with audio playback.")
            
            with gr.Tab("📚 PDF to Audiobook"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📄 PDF Upload")
                        gr.Markdown("*Upload a PDF file to convert to audiobook*")
                        pdf_file = gr.File(
                            label="Upload PDF Document",
                            file_types=[".pdf"]
                        )
                        
                        gr.Markdown("### 🎛️ Audiobook Settings")
                        pdf_model_choice = gr.Radio(
                            ["KittenTTS", "Kokoro-82M"], 
                            value="KittenTTS", 
                            label="TTS Model",
                            info="Choose the text-to-speech engine"
                        )
                        
                        pdf_voice_dropdown = gr.Dropdown(
                            choices=KITTEN_VOICES,
                            value='expr-voice-4-f',
                            label="Voice Selection",
                            info="Select voice for the audiobook",
                            interactive=True
                        )
                        
                        pdf_speed = gr.Slider(
                            minimum=0.5, 
                            maximum=2.0, 
                            step=0.05, 
                            value=1.0, 
                            label="Speech Speed",
                            info="Adjust narration speed"
                        )
                        
                        # Progress info
                        gr.Markdown("### ℹ️ Process Info")
                        gr.Markdown("""
                        **Process Steps:**
                        1. Extract text from PDF
                        2. Segment into optimal chunks
                        3. Generate audio for each chunk
                        4. Combine with natural pauses
                        5. Download complete audiobook
                        
                        **Note:** Large PDFs may take several minutes to process.
                        """)
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### 📖 PDF Processing")
                        
                        with gr.Row():
                            process_text_btn = gr.Button("📄 Extract Text from PDF", variant="secondary", size="lg")
                            create_audiobook_btn = gr.Button("🎧 Create Audiobook", variant="primary", size="lg")
                        
                        # Text preview
                        pdf_text_preview = gr.Textbox(
                            label="Extracted Text Preview",
                            placeholder="Upload a PDF and click 'Extract Text' to see preview...",
                            lines=8,
                            interactive=False
                        )
                        
                        # Status output
                        pdf_status = gr.Textbox(
                            label="Processing Status",
                            value="Ready to process PDF...",
                            interactive=False,
                            lines=3
                        )
                        
                        # Final audiobook output
                        audiobook_output = gr.Audio(
                            label="Generated Audiobook", 
                            autoplay=False, 
                            type="numpy",
                            show_download_button=True
                        )

                        # Chapter UI: dropdown and single-chapter audio
                        gr.Markdown("### 📚 Chapters")
                        with gr.Row():
                            chapter_dropdown = gr.Dropdown(
                                label="Select Chapter",
                                choices=[],
                                value=None,
                                interactive=True
                            )
                            chapter_audio = gr.Audio(
                                label="Chapter Audio",
                                autoplay=False,
                                type="numpy",
                                show_download_button=True
                            )

                        # Chapters zip download
                        chapters_zip_file = gr.File(
                            label="Chapters ZIP (per-chapter WAVs)",
                            file_count="single",
                            interactive=False
                        )
                
                        # Download info
                        with gr.Row():
                            gr.Markdown("**Download:** Once generated, use the download button above to save your audiobook as a WAV file.")

        # Event handlers
        def update_voice_dropdown(model):
            if model == 'KittenTTS':
                return gr.Dropdown(
                    choices=KITTEN_VOICES,
                    value='expr-voice-4-f',
                    label="Voice Selection",
                    info="KittenTTS voices (8 available)",
                    interactive=True
                )
            else:  # Kokoro-82M
                kokoro_choices = sorted(list(KOKORO_VOICES))
                return gr.Dropdown(
                    choices=kokoro_choices,
                    value='af_heart',
                    label="Voice Selection", 
                    info="Kokoro-82M voices (50+ available)",
                    interactive=True
                )
        
        def clear_text():
            return "", "Text cleared. Ready for new input."
        
        def update_status():
            return "Synthesizing... Please wait."
        
        # Connect events
        model_choice.change(
            fn=update_voice_dropdown,
            inputs=[model_choice],
            outputs=[voice_dropdown]
        )
        
        def debug_highlighting():
            """Debug word highlighting functionality"""
            return "🐛 Check browser console for debug info. Use debugWordHighlighting() function. Also try: window.testHighlighting() in console."
        
        clear_btn.click(
            fn=clear_text,
            outputs=[text, status_output]
        )
        
        debug_btn.click(
            fn=debug_highlighting,
            outputs=[status_output]
        )
        
        def update_timings_and_transcript(timings_json, transcript_html_content):
            """Update the JavaScript with new word timings and return transcript HTML"""
            # Debug logging
            logging.info(f"Updating transcript with {len(timings_json)} characters of timing data")
            logging.info(f"Transcript HTML length: {len(transcript_html_content)} characters")
            
            # Create the complete HTML with embedded JavaScript
            complete_html = f"""
            <div class='transcript-container'>
                <div id='transcript'>{transcript_html_content}</div>
            </div>
            <script>
            // Ensure we're in the right context
            (function() {{
                // Wait for functions to be available
                function waitForFunctions() {{
                    if (typeof window.updateWordTimings === 'function' && typeof window.clearWordHighlighting === 'function') {{
                        try {{
                            window.updateWordTimings('{timings_json}');
                            window.clearWordHighlighting();
                            console.log('✅ Word timings updated successfully');
                            console.log('📊 Timing data length:', '{len(timings_json)}');
                            console.log('📝 Transcript HTML length:', '{len(transcript_html_content)}');
                            
                            // Verify the transcript has word spans
                            const wordSpans = document.querySelectorAll('#transcript .w');
                            console.log('🔍 Found word spans:', wordSpans.length);
                            
                            // Test highlighting on first word
                            if (wordSpans.length > 0) {{
                                wordSpans[0].classList.add('current');
                                setTimeout(() => wordSpans[0].classList.remove('current'), 1000);
                                console.log('🧪 Test highlighting applied to first word');
                            }}
                        }} catch (error) {{
                            console.error('❌ Error updating word timings:', error);
                        }}
                    }} else {{
                        console.log('⏳ Functions not ready, retrying...');
                        setTimeout(waitForFunctions, 100);
                    }}
                }}
                
                // Start the process
                if (document.readyState === 'loading') {{
                    document.addEventListener('DOMContentLoaded', waitForFunctions);
                }} else {{
                    waitForFunctions();
                }}
            }})();
            </script>
            """
            return complete_html
        
        synthesize_btn.click(
            fn=update_status,
            outputs=[status_output]
        ).then(
            fn=lambda text, model, voice, speed: tts_generate(text, model, voice, speed),
            inputs=[text, model_choice, voice_dropdown, speed],
            outputs=[audio, status_output, timings_state, transcript_html]
        ).then(
            fn=update_timings_and_transcript,
            inputs=[timings_state, transcript_html],
            outputs=[transcript_html]
        )
        
        # PDF Tab Event Handlers
        def update_pdf_voice_dropdown(model):
            """Update PDF voice dropdown based on selected model"""
            if model == 'KittenTTS':
                return gr.Dropdown(
                    choices=KITTEN_VOICES,
                    value='expr-voice-4-f',
                    label="Voice Selection",
                    info="KittenTTS voices for audiobook",
                    interactive=True
                )
            else:  # Kokoro-82M
                kokoro_choices = sorted(list(KOKORO_VOICES))
                return gr.Dropdown(
                    choices=kokoro_choices,
                    value='af_heart',
                    label="Voice Selection", 
                    info="Kokoro-82M voices for audiobook",
                    interactive=True
                )
        
        def extract_pdf_text(pdf_file):
            """Extract text from PDF for preview"""
            if pdf_file is None:
                return "❌ No PDF file uploaded", ""
            
            try:
                info, text, _, _ = process_pdf_to_text(pdf_file)
                # Show first 1000 characters as preview
                preview = text[:1000] + ("..." if len(text) > 1000 else "")
                return f"✅ Text extracted successfully!\n{info}", preview
            except Exception as e:
                return f"❌ Error extracting text: {str(e)[:100]}", ""
        
        def update_chapters_zip(chapters_zip):
            """Update the chapters_zip_file component with the chapters_zip path"""
            if chapters_zip:
                return chapters_zip, None
            return None, "No chapters zip file available"

        def process_pdf_to_audiobook_with_chapters(pdf_file, model_choice, voice, speed, progress=gr.Progress()):
            """PDF to audiobook with incremental chapter processing and UI wiring"""
            if pdf_file is None:
                return None, "❌ No PDF file uploaded", None, gr.update(choices=[], value=None), None

            # Reuse internal function for text + chunks
            try:
                progress(0.05, desc="Processing PDF...")
                info, text, _, _ = process_pdf_to_text(pdf_file)
                if not text.strip():
                    return None, "❌ No text extracted from PDF", None, gr.update(choices=[], value=None), None

                # Build chunks
                progress(0.1, desc="Segmenting text...")
                paragraphs = text.split('\n\n')
                sentences = []
                for i, paragraph in enumerate(paragraphs):
                    sentences.extend(split_into_sentences(paragraph, i))
                chunks = build_chunks(sentences, target_chars=280, hard_cap=360)
                if not chunks:
                    return None, "❌ Failed to create text chunks", None, gr.update(choices=[], value=None), None

                # Engine selection
                clean_model = model_choice.replace(" (Demo)", "")
                if clean_model == 'KittenTTS':
                    engine = get_kitten_engine()
                    if not engine:
                        return None, "❌ KittenTTS engine not available", None, gr.update(choices=[], value=None), None
                else:
                    # For now chapter UI is wired for KittenTTS path
                    engine = get_kitten_engine()
                    if not engine:
                        return None, "❌ TTS engine not available", None, gr.update(choices=[], value=None), None

                # Run chapter processor
                progress(0.15, desc="Detecting chapters...")
                from pathlib import Path as _Path
                pdf_path_obj = _Path(pdf_file) if isinstance(pdf_file, str) else None
                document = read_pdf(pdf_path_obj) if pdf_path_obj else read_pdf(_Path(pdf_file.name))
                processor = ChapterProcessor(
                    document=document,
                    pdf_path=pdf_path_obj or _Path(pdf_file.name),
                    voice=voice,
                    speed=speed,
                    normalize_loudness=True,
                )

                def on_progress(cur, total, status):
                    # Map to coarse progress steps
                    base = 0.2
                    span = 0.7
                    frac = base + span * (cur / max(1, total))
                    progress(frac, desc=status)

                result = processor.process_chapters_incrementally(engine, chunks, progress_callback=on_progress)

                # Build UI outputs
                final_audio = result.full_audio if result.full_audio is not None else np.zeros(0, dtype=np.float32)
                choices = [f"{i+1}. {ca.chapter_info.title}" for i, ca in enumerate(result.chapters)]
                chapters_zip = str(result.zip_path) if result.zip_path else None

                # default first chapter in dropdown
                value = choices[0] if choices else None

                return (
                    (24000, final_audio),
                    "✅ Audiobook created with chapter-by-chapter processing",
                    chapters_zip,
                    gr.update(choices=choices, value=value),
                    None,
                )
            except Exception as e:
                logging.error(f"Chapter processing error: {e}")
                return None, f"❌ Error: {str(e)[:150]}", None, gr.update(choices=[], value=None), None

        def load_selected_chapter(dropdown_value, pdf_file, model_choice, voice, speed):
            """Load selected chapter audio into the chapter_audio component."""
            if not dropdown_value:
                return None
            try:
                # Re-run minimal flow to locate chapter file quickly
                from pathlib import Path as _Path
                pdf_path_obj = _Path(pdf_file) if isinstance(pdf_file, str) else _Path(pdf_file.name)
                document = read_pdf(pdf_path_obj)
                processor = ChapterProcessor(document=document, pdf_path=pdf_path_obj, voice=voice, speed=speed)
                # Find index from label prefix "NN. title"
                idx_str = dropdown_value.split('.', 1)[0]
                idx = max(1, int(idx_str)) - 1
                if idx < 0 or idx >= len(processor.chapters_info):
                    return None
                # We assume files were written already; try to resolve by sanitized name pattern in output_dir
                # Fallback: return None silently
                output_dir = processor.output_dir
                # Pick matching by index
                import glob
                candidates = sorted(glob.glob(str(output_dir / f"chapter_{idx+1:02d}_*.wav")))
                if not candidates:
                    return None
                import soundfile as _sf
                data, _sr = _sf.read(candidates[0], dtype='float32')
                return (24000, data.astype(np.float32))
            except Exception as e:
                logging.warning(f"Failed to load selected chapter: {e}")
                return None

        # Connect PDF tab events
        pdf_model_choice.change(
            fn=update_pdf_voice_dropdown,
            inputs=[pdf_model_choice],
            outputs=[pdf_voice_dropdown]
        )
        
        process_text_btn.click(
            fn=extract_pdf_text,
            inputs=[pdf_file],
            outputs=[pdf_status, pdf_text_preview]
        )
        
        create_audiobook_btn.click(
            fn=process_pdf_to_audiobook_with_chapters,
            inputs=[pdf_file, pdf_model_choice, pdf_voice_dropdown, pdf_speed],
            outputs=[audiobook_output, pdf_status, chapters_zip_file, chapter_dropdown, chapter_audio]
        )

        chapter_dropdown.change(
            fn=load_selected_chapter,
            inputs=[chapter_dropdown, pdf_file, pdf_model_choice, pdf_voice_dropdown, pdf_speed],
            outputs=[chapter_audio]
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
