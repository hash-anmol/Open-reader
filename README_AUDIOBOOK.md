# 📚 Kitten TTS Audiobook Generator

A powerful PDF-to-audiobook converter using advanced text-to-speech technology. Convert any PDF document into high-quality audiobooks with natural-sounding voices.

## ✨ Features

### 🎙️ **Text-to-Speech**
- **KittenTTS**: Lightweight, CPU-only TTS engine (25MB)
- **Kokoro-82M**: High-quality TTS with multiple voice options
- **Real-time word highlighting** synchronized with audio playback
- **Voice selection** with 8+ voice options
- **Speed control** from 0.5x to 2.5x
- **Smart fallback strategies** for robust synthesis

### 📚 **PDF Audiobook Conversion**
- **PDF Upload**: Drag and drop any PDF document
- **Automatic text extraction** with cleaning and normalization
- **Intelligent chunking** optimized for TTS (350 char target, 400 max)
- **Progress tracking** with real-time status updates
- **Audio assembly** with appropriate pauses between sentences and paragraphs
- **Download support** for complete audiobook files
- **Chapter detection** (future feature)

### 🎯 **Advanced Features**
- **Tabbed interface** for different modes
- **Status monitoring** with detailed feedback
- **Error handling** with graceful fallbacks
- **Memory efficient** processing for large documents
- **High-quality audio** at 24kHz sample rate

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Activate virtual environment
source venv/bin/activate

# Install required packages (if not already installed)
pip install -r requirements.txt
```

### 2. Launch Application
```bash
# Simple launch
python launch_audiobook.py

# Or directly run the UI
python src/kitten_audiobook/ui/app.py
```

### 3. Open Web Interface
- Open your browser to `http://127.0.0.1:7860`
- The application will automatically open two tabs:
  - **🎙️ Text-to-Speech**: Convert text directly
  - **📚 PDF Audiobook**: Convert PDF documents

## 📖 Usage Guide

### Text-to-Speech Mode
1. **Select TTS Model**: Choose between KittenTTS or Kokoro-82M
2. **Choose Voice**: Pick from available voices for your selected model
3. **Set Speed**: Adjust speech speed (0.5x - 2.5x)
4. **Enter Text**: Type or paste your text
5. **Synthesize**: Click "🎙️ Synthesize Audio"
6. **Listen & Download**: Play audio with word highlighting, download when ready

### PDF Audiobook Mode
1. **Upload PDF**: Click "📁 Upload PDF Document" and select your file
2. **Configure Settings**:
   - Select TTS model (KittenTTS or Kokoro-82M)
   - Choose voice
   - Set speech speed
3. **Process PDF**: Click "📄 Process PDF" to extract and preview text
4. **Create Audiobook**: Click "🎧 Create Audiobook" to start conversion
5. **Monitor Progress**: Watch real-time progress updates
6. **Download**: Save your complete audiobook when finished

## 🎵 Voice Options

### KittenTTS Voices
- `expr-voice-2-f/m`: Female/Male voice (style 2)
- `expr-voice-3-f/m`: Female/Male voice (style 3)
- `expr-voice-4-f/m`: Female/Male voice (style 4)
- `expr-voice-5-f/m`: Female/Male voice (style 5)

### Kokoro-82M Voices (if available)
- **American Female**: `af_alloy`, `af_bella`, `af_heart`, `af_nova`, etc.
- **American Male**: `am_adam`, `am_echo`, `am_liam`, `am_onyx`, etc.
- **British Female**: `bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily`
- **British Male**: `bm_daniel`, `bm_george`, `bm_lewis`, `bm_fable`

## 📁 Project Structure

```
kitten_tts/
├── src/kitten_audiobook/
│   ├── ui/
│   │   └── app.py              # Main Gradio interface
│   ├── ingestion/
│   │   ├── pdf_reader.py       # PDF text extraction
│   │   └── structure.py        # Document data models
│   ├── text/
│   │   ├── cleaning.py         # Text normalization
│   │   ├── segmentation.py     # Sentence segmentation
│   │   └── chunking.py         # TTS-optimized chunking
│   ├── tts/
│   │   ├── engine.py           # TTS engine wrapper
│   │   └── synthesis.py        # Audio synthesis
│   └── utils/
│       ├── io.py               # File utilities
│       └── logging.py          # Logging setup
├── data/
│   └── cache/                  # TTS synthesis cache
├── output/
│   └── test_mp3/               # Audio output directory
├── launch_audiobook.py         # Application launcher
├── test_document.pdf           # Sample PDF for testing
├── requirements.txt            # Python dependencies
└── taskpad.md                  # Development roadmap
```

## 🔧 Technical Details

### Text Processing Pipeline
1. **PDF Extraction**: Uses PyMuPDF for robust text extraction
2. **Cleaning**: Unicode normalization, character filtering, smart quote conversion
3. **Segmentation**: Sentence detection with abbreviation handling
4. **Chunking**: Optimized 350-character chunks with sentence boundary respect

### Audio Processing
- **Sample Rate**: 24kHz mono
- **Silence Insertion**: 200ms between chunks, 800ms between paragraphs
- **Format Support**: MP3, WAV download options
- **Caching**: Per-chunk synthesis cache for efficiency

### Performance Optimizations
- **Sequential Processing**: Memory-efficient chunk-by-chunk synthesis
- **Progress Tracking**: Real-time updates during conversion
- **Error Recovery**: Graceful handling of failed chunks
- **Fallback Strategies**: Multiple synthesis approaches

## 🧪 Testing

Use the included test document:
```bash
# The application includes test_document.pdf
# Upload this file in the PDF Audiobook tab to test functionality
```

Create your own test content:
```bash
# Generate a new test PDF
python test_audiobook.py
```

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'gradio'"**
   - Solution: Run `pip install gradio` in your virtual environment

2. **"No module named 'fitz'"**
   - Solution: Run `pip install pymupdf`

3. **"Kokoro TTS not available"**
   - This is normal - Kokoro is optional
   - KittenTTS works perfectly for all functionality

4. **PDF processing fails**
   - Ensure PDF is not password-protected
   - Try with the included `test_document.pdf`
   - Check PDF contains readable text (not just images)

5. **Audio synthesis fails**
   - Check that text contains readable content
   - Try reducing text length for testing
   - Verify virtual environment is activated

### Performance Tips
- **Large PDFs**: Process in batches or use shorter documents for testing
- **Memory Usage**: Close other applications for large conversions
- **Speed**: KittenTTS is faster than Kokoro but with fewer voice options

## 🔮 Future Features (Planned)

### Chapter-Based Processing
- Automatic chapter detection using PDF bookmarks
- Individual audio files per chapter
- Chapter navigation and metadata
- Combined audiobook with chapter markers

### Advanced Options
- Voice cloning capabilities
- Emotion and style controls
- Multiple output formats (M4B, etc.)
- Batch processing for multiple PDFs

### Enhanced UI
- Dark mode theme
- Advanced audio controls
- Bookmark and resume functionality
- Cloud storage integration

## 📝 Development

Built with:
- **KittenTTS**: Lightweight TTS engine
- **Gradio**: Web interface framework
- **PyMuPDF**: PDF processing
- **NumPy**: Audio array processing
- **Python 3.10+**: Core language

For development details, see `taskpad.md`.

## 🙏 Credits

- **KittenML**: For the excellent KittenTTS engine
- **Gradio**: For the amazing web interface framework
- **PyMuPDF**: For robust PDF processing
- **Kokoro TTS**: For high-quality voice synthesis

---

**Ready to convert your PDFs to audiobooks? Launch the application and start creating!** 🎧✨