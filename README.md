# KittenTTS Audiobook Generator

A comprehensive text-to-speech application that converts text and PDF documents into high-quality audiobooks using KittenTTS and Kokoro-82M engines.

I have added an entire guide for non-coders to install and set this up. But if you are a coder and want to contribute to this project, feel free to do so!

Infact I would really appriciate if the community can help me improve this system with more features and a more robust voice TTS experience.

## 🎯 Features

- 🎙️ **Dual TTS Engines**: KittenTTS (lightweight, CPU-only) and Kokoro-82M (high-quality)
- 📚 **PDF to Audiobook**: Convert entire PDF documents into narrated audiobooks
- 🗣️ **Multiple Voices**: 8+ KittenTTS voices and 50+ Kokoro voices
- 📖 **Word Highlighting**: Real-time word highlighting synchronized with audio playback
- 🎛️ **Speed Control**: Adjustable speech speed from 0.5x to 2.5x
- 📄 **Chapter Detection**: Automatic chapter segmentation for long documents
- 💾 **Audio Export**: Download as WAV files or chapter-by-chapter ZIP archives
- 🌐 **Web Interface**: User-friendly Gradio interface

## 🚀 Complete Setup Guide for Beginners

> **👋 New to coding?** Don't worry! This guide will walk you through everything step-by-step. No prior programming experience needed.

### 📋 What You'll Need First

Before we start, you need to install some basic tools on your computer:

#### Step A: Install Git (for downloading the code)

**Windows Users:**
1. Go to https://git-scm.com/download/win
2. Download Git for Windows  
3. Run the installer with default settings
4. After installation, you can right-click anywhere and see "Git Bash" - that's how you'll know it worked!

**Mac Users:**
1. Open "Terminal" (press Cmd+Space, type "terminal", press Enter)
2. Type: `git --version`
3. If Git is already installed, you'll see a version number
4. If not, macOS will prompt you to install Xcode Command Line Tools - click "Install"

**Linux Users:**
```bash
# Ubuntu/Debian:
sudo apt update && sudo apt install git

# CentOS/RHEL/Fedora:  
sudo dnf install git
```

#### Step B: Install Python (the programming language this app uses)

**⚠️ CRITICAL:** You need Python version 3.9, 3.10, 3.11, or 3.12. **NOT 3.13 or higher** - it won't work!

**Windows Users:**
1. Go to https://www.python.org/downloads/
2. **Don't click the big download button!** Scroll down to find Python 3.12.x
3. Download and run the installer
4. **IMPORTANT:** Check "Add Python to PATH" during installation
5. Test: Open Command Prompt (Win+R, type `cmd`) and type `python --version`

**Mac Users:**
1. First check what you have: Open Terminal and type `python3 --version`
2. If you see 3.13+, you need to install 3.12 instead:
   ```bash
   # Install Homebrew first (if you don't have it):
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   # Then install Python 3.12:
   brew install python@3.12
   ```

**Linux Users:**
```bash
# Ubuntu/Debian:
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev

# CentOS/RHEL/Fedora:
sudo dnf install python3.12 python3.12-devel
```

---

## 🔧 Installing the TTS Application

Now that you have Git and Python installed, let's get the application running!

### Step 1: Download the Application Code

**What is "cloning"?** It means downloading all the code files to your computer.

**For Windows Users:**
1. Right-click on your Desktop  
2. Select "Git Bash Here"
3. In the black window that opens, copy and paste this command:
   ```bash
   git clone https://github.com/your-username/kitten_tts.git
   ```
4. Press Enter and wait for it to download
5. Type: `cd kitten_tts` and press Enter

**For Mac/Linux Users:**
1. Open Terminal
2. Navigate to where you want the app (like Desktop):
   ```bash
   cd Desktop
   ```
3. Download the code:
   ```bash
   git clone https://github.com/your-username/kitten_tts.git
   cd kitten_tts
   ```

**✅ Success Check:** You should now see files like `README.md`, `requirements.txt` when you type `ls` (Mac/Linux) or `dir` (Windows)

### Step 2: Create a Safe Space for the App (Virtual Environment)

**What's a virtual environment?** Think of it as a separate folder where this app keeps all its specific requirements, so it doesn't mess with other software on your computer.

**Check Your Python Version First:**

**Windows Users:** 
```bash
python --version
```

**Mac/Linux Users:**
```bash
python3 --version
```

**You should see something like "Python 3.11.5" or "Python 3.12.1"**

**If you see Python 3.13+ or get an error:**
- Go back to Step B above and install Python 3.12
- Make sure you can run the version check successfully before continuing

**Create the Virtual Environment:**

**Windows Users:**
```bash
# If you have Python 3.12 installed:
py -3.12 -m venv venv

# Or if your default Python is correct (3.9-3.12):
python -m venv venv
```

**Mac Users:**
```bash
# If you installed Python 3.12 with Homebrew:
python3.12 -m venv venv

# Or if your default Python is correct:
python3 -m venv venv
```

**Linux Users:**
```bash
# If you installed Python 3.12:
python3.12 -m venv venv

# Or if your default Python is correct:
python3 -m venv venv
```

**✅ Success Check:** You should now see a new folder called `venv` in your kitten_tts directory

### Step 3: Activate Your Virtual Environment

**What does "activate" mean?** It tells your computer to use the Python and packages from this special folder instead of your system's default ones.

**Windows Users:**
1. In your Git Bash or Command Prompt (make sure you're in the kitten_tts folder):
   ```bash
   venv\Scripts\activate
   ```
2. **Success:** Your command prompt should now start with `(venv)` - this means it worked!

**Mac/Linux Users:**
1. In Terminal (make sure you're in the kitten_tts folder):
   ```bash
   source venv/bin/activate
   ```  
2. **Success:** Your command prompt should now start with `(venv)` - this means it worked!

**Double-Check Your Python Version:**
```bash
python --version
```
**You should see Python 3.9, 3.10, 3.11, or 3.12. If you see 3.13+, something went wrong - start over with Step 2.**

### Step 4: Install the App's Requirements

**What are dependencies?** These are other pieces of software that our app needs to work - like ingredients for a recipe.

**Copy and paste these commands one by one:**

1. **First, upgrade the installer tool:**
   ```bash
   pip install --upgrade pip
   ```
   *Wait for this to finish (you'll see "Successfully upgraded pip")*

2. **Install the main requirements:**
   ```bash
   pip install -r requirements_minimal.txt
   ```
   *This will take 2-5 minutes and show lots of text. That's normal! Wait for it to complete.*

3. **Install the high-quality voice engine (optional but recommended):**
   ```bash
   pip install kokoro
   ```
   *This might take another 2-3 minutes*

**✅ Success Check:** If you see "Successfully installed" messages without any red error text, you're good to go!

### Step 5: Download Language Processing (Optional but Recommended)

**What's this for?** It helps the app understand English text better for more natural speech.

```bash
python -m spacy download en_core_web_sm
```
*This downloads about 50MB and might take 1-2 minutes*

### Step 6: Start the Application! 🚀

**The moment of truth! Let's run the app:**

```bash
python -m src.kitten_audiobook.ui.app
```

**What you should see:**
- Lots of text will appear (don't worry, this is normal!)
- Look for these important lines:
  ```
  INFO:root:✅ Kokoro TTS loaded successfully  
  INFO:root:KittenTTS engine initialized
  * Running on local URL: http://127.0.0.1:7860
  ```

**⚠️ Don't close this window!** The app runs from here. Minimize it if needed.

### Step 7: Use the App in Your Web Browser

1. **Open your web browser** (Chrome, Firefox, Safari, Edge - any will work)
2. **Go to this address:** `http://localhost:7860`
3. **You should see a beautiful interface** with two tabs:
   - "🎙️ Text-to-Speech" - Convert any text to speech
   - "📚 PDF to Audiobook" - Convert PDF documents to audiobooks

### 🎉 Test Your Setup

**Let's make sure everything works:**

1. **Click on the "🎙️ Text-to-Speech" tab**
2. **In the text box, type:** "Hello world, this is my new text to speech application!"
3. **Click the "🎙️ Synthesize Audio" button**
4. **Wait 5-10 seconds** (be patient!)
5. **You should hear a voice saying your text!** 

**🎊 Congratulations! You've successfully set up the TTS application!**

### ✅ Verify Installation

After starting the app, you should see:
```bash
* Running on local URL:  http://127.0.0.1:7860
INFO:root:✅ Kokoro TTS loaded successfully  
INFO:root:KittenTTS engine initialized
```

**Test the setup:**
1. Open http://localhost:7860 in your browser
2. Go to "🎙️ Text-to-Speech" tab
3. Enter "Hello, this is a test" in the text box
4. Click "🎙️ Synthesize Audio"
5. You should hear generated speech - **setup complete!** 🎉

## 📋 System Requirements

### Minimum Requirements
- **RAM**: 4GB (8GB recommended for large PDFs)
- **Storage**: 1GB free space
- **CPU**: Any modern CPU (GPU optional for Kokoro)
- **Python**: 3.9 - 3.12

### Recommended Setup
- **RAM**: 8GB+ 
- **Storage**: 5GB+ (for voice models and output files)
- **CPU**: Multi-core processor
- **GPU**: CUDA-compatible GPU (for Kokoro acceleration)

## 🎮 Usage Guide

### Text-to-Speech Tab

1. **Select TTS Model**: Choose between KittenTTS (fast, CPU) or Kokoro-82M (high-quality)
2. **Pick a Voice**: Select from available voices in the dropdown
3. **Adjust Speed**: Use slider to control speech rate (0.5x - 2.5x)
4. **Enter Text**: Type or paste your text in the input field
5. **Generate Audio**: Click "🎙️ Synthesize Audio" 
6. **Play & Download**: Use the audio player to listen and download

### PDF-to-Audiobook Tab

1. **Upload PDF**: Click "Upload PDF Document" and select your file
2. **Extract Text**: Click "📄 Extract Text from PDF" to preview content
3. **Configure Settings**: Choose TTS model, voice, and speed
4. **Create Audiobook**: Click "🎧 Create Audiobook" (may take several minutes)
5. **Download Results**: 
   - Full audiobook as single WAV file
   - Individual chapters as ZIP archive
   - Chapter-by-chapter audio player

---

## 🎮 How to Use Your New TTS Application

### 🎙️ Converting Text to Speech

1. **Choose Your Voice Engine:**
   - **KittenTTS**: Faster, works on any computer, smaller download
   - **Kokoro-82M**: Higher quality, more realistic voices, larger download

2. **Select a Voice:**
   - For KittenTTS: Try `expr-voice-4-f` (female) or `expr-voice-4-m` (male)
   - For Kokoro: Try `af_heart` (American female) or `am_adam` (American male)

3. **Adjust Speed:** Use the slider (1.0 = normal speed, 0.5 = slower, 2.0 = faster)

4. **Enter Your Text:** Type anything you want - quotes, stories, emails, etc.

5. **Generate and Download:** Click synthesize, wait, then download your audio file!

### 📚 Converting PDFs to Audiobooks

1. **Upload Your PDF:** Click "Upload PDF Document" and select any PDF file
2. **Extract Text First:** Click "📄 Extract Text from PDF" to preview what will be read
3. **Choose Settings:** Pick voice and speed like above  
4. **Create Audiobook:** Click "🎧 Create Audiobook" (this takes longer - be patient!)
5. **Download Options:**
   - Single audiobook file (WAV format)
   - Chapter-by-chapter ZIP file
   - Individual chapter player

**💡 Tips:**
- **Large PDFs**: May take 10-30 minutes depending on size
- **Chapter Detection**: The app automatically finds chapters in your PDF
- **Quality vs Speed**: Kokoro sounds better but takes longer

---

## 🆘 Help! Something Went Wrong

### "I Can't Install Python 3.12"

**Windows Users:**
- Make sure you downloaded from python.org (not Microsoft Store)
- Run installer as Administrator (right-click installer → "Run as administrator")
- Restart your computer after installation

**Mac Users:**
- If Homebrew installation fails, try the direct download from python.org
- You might need to allow installation in System Preferences → Security

**Linux Users:**
- Try adding the deadsnakes PPA for Ubuntu: `sudo add-apt-repository ppa:deadsnakes/ppa`

### "Command Not Found" or "python is not recognized"

**This means Python isn't in your PATH:**
- **Windows**: Reinstall Python and check "Add Python to PATH" 
- **Mac**: Try `python3` instead of `python`
- **All**: Restart your terminal/command prompt after installing Python

### "Git Command Not Found"

- **Windows**: Make sure you installed Git from git-scm.com
- **Mac**: Install Xcode Command Line Tools: `xcode-select --install`
- **Linux**: Install git with your package manager

### "Permission Denied" Errors

- **Don't use `sudo` with pip** - always work in your virtual environment
- On Mac/Linux, make sure you own the folder: `sudo chown -R $USER:$USER ~/kitten_tts`

### "The App Starts But No Sound"

1. **Check your browser audio permissions** (click the speaker icon in the URL bar)
2. **Try a different browser** (Chrome or Firefox work best)
3. **Check your computer's volume** and speaker settings
4. **Try different voice/model combinations**

### "PDF Processing Fails"

- **Make sure your PDF has text** (not just scanned images)
- **Try a smaller PDF first** to test the system
- **Check the extracted text preview** - if it's gibberish, the PDF might be corrupted

### "Virtual Environment Issues"

**Start completely fresh:**
```bash
# Delete the old environment
rm -rf venv  # Mac/Linux
rmdir /s venv  # Windows

# Create a new one with the right Python version
python3.12 -m venv venv  # Mac/Linux  
py -3.12 -m venv venv    # Windows

# Activate and try again
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### "Still Stuck? Try This Step-by-Step Reset"

1. **Close everything** (terminal, browser, etc.)
2. **Delete the whole kitten_tts folder**
3. **Start over from Step 1** of the installation guide
4. **Follow each step exactly** without skipping

---

## 💡 Frequently Asked Questions

**Q: Is this free?**  
A: Yes! Completely free and open source.

**Q: Do I need internet after installation?**  
A: Only for the first run to download voice models. After that, it works offline.

**Q: Can I use my own voice?**  
A: Not currently, but the app includes many different voice options.

**Q: What file formats can I convert?**  
A: PDFs for documents, and any text for speech. Output is WAV audio format.

**Q: How long can my text be?**  
A: No limit! The app handles everything from single sentences to entire books.

**Q: Can I stop the app and restart later?**  
A: Yes! Just close the terminal window to stop, and run Step 6 again to restart.

---

## 🛠️ Technical Troubleshooting (Advanced Users)

### Common Issues

**1. Python Version Compatibility Issues**
```bash
# Check your Python version first
python --version

# If you see Python 3.13+, you need to use a compatible version:
# macOS: brew install python@3.12 && python3.12 -m venv venv
# Linux: sudo apt install python3.12-venv && python3.12 -m venv venv
# Windows: Download Python 3.12 from python.org

# Always verify after creating venv:
source venv/bin/activate  # or venv\Scripts\activate on Windows  
python --version  # Should show 3.9.x - 3.12.x
```

**2. ModuleNotFoundError: No module named 'gradio'**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements_minimal.txt
```

**3. KittenTTS Engine Not Available**
- With correct Python version (3.9-3.12), KittenTTS should install properly
- App will work with Kokoro-82M if KittenTTS has issues
- Try: `pip install --upgrade pip` then reinstall dependencies

**4. Kokoro Installation Still Fails**
- Ensure you're using Python 3.9-3.12 (not 3.13+)  
- Try: `pip install --upgrade setuptools wheel` first
- If still failing, app works with KittenTTS only

**5. Dependency Conflicts**
```bash
# Start fresh with correct Python version
rm -rf venv  # Remove old virtual environment
python3.12 -m venv venv  # Create with compatible Python
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements_minimal.txt
```

**6. PDF Processing Errors**
```bash
# Ensure PyMuPDF is properly installed
pip install --upgrade pymupdf
```

**7. Audio Playback Issues**
- Check browser audio permissions
- Try different browsers (Chrome/Firefox recommended)  
- Ensure firewall isn't blocking localhost:7860

**8. Memory Issues with Large PDFs**
- Split large PDFs into smaller sections
- Increase system RAM or use swap space
- Process chapters individually using the chapter dropdown

### Development Setup

**For developers wanting to modify the code:**

```bash
# Install development dependencies
pip install -r requirements.txt  # Full requirements (may have conflicts)

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# Run tests
python -m pytest tests/
```

## 📁 Project Structure

```
kitten_tts/
├── src/kitten_audiobook/
│   ├── ui/app.py              # Main Gradio interface
│   ├── tts/
│   │   ├── engine.py          # TTS engine wrapper
│   │   └── synthesis.py       # Audio synthesis logic
│   ├── ingestion/
│   │   ├── pdf_reader.py      # PDF text extraction
│   │   └── chapter_detection.py # Chapter segmentation
│   └── utils/
├── requirements.txt           # Full dependencies (may have conflicts)
├── requirements_minimal.txt   # Core dependencies (recommended)
├── data/                      # Cached audio files
├── output/                    # Generated audiobooks
└── tests/                     # Unit tests
```

## 🎤 Available Voices

### KittenTTS Voices (8 total)
- `expr-voice-2-m/f`: Expression voice 2 (Male/Female)
- `expr-voice-3-m/f`: Expression voice 3 (Male/Female) 
- `expr-voice-4-m/f`: Expression voice 4 (Male/Female)
- `expr-voice-5-m/f`: Expression voice 5 (Male/Female)

### Kokoro-82M Voices (50+ total)
- **American Female**: `af_alloy`, `af_heart`, `af_nova`, `af_sarah`, `af_sky`, etc.
- **American Male**: `am_adam`, `am_echo`, `am_liam`, `am_michael`, etc.
- **British Female**: `bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily`
- **British Male**: `bm_daniel`, `bm_george`, `bm_lewis`, etc.

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set cache directory
export KITTEN_CACHE_DIR=/path/to/cache

# Optional: Set output directory  
export KITTEN_OUTPUT_DIR=/path/to/output

# Optional: Set Gradio server port
export GRADIO_SERVER_PORT=7860
```

### Voice Model Downloads
Voice models are downloaded automatically on first use. Large models may take time to download initially.

## 📜 License

This project is open source. Please check individual dependencies for their respective licenses:
- KittenTTS: Check KittenML/KittenTTS repository
- Kokoro-82M: Check Kokoro TTS repository  
- Gradio: Apache 2.0 License

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with clear description

## 📞 Support

- **Issues**: Report bugs and feature requests on GitHub Issues
- **Documentation**: Check README and code comments
- **Community**: Join discussions in GitHub Discussions

## 🔄 Updates

To update to the latest version:

```bash
git pull origin main
pip install -r requirements_minimal.txt --upgrade
```

---

**🎉 Enjoy creating audiobooks with KittenTTS!** 

*Convert any text or PDF into professional-quality narrated audio with just a few clicks.*