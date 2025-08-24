# 🎧 Kitten TTS - PDF to Audiobook Setup Guide

## 🚨 **What Was Wrong & What We Fixed**

### **Critical Issues Identified:**

1. **🐍 Python Version Incompatibility**
   - **Problem**: Python 3.13 was incompatible with Kokoro TTS (requires Python <3.13)
   - **Solution**: Downgraded to Python 3.12.11 for full compatibility

2. **🔧 TTS Engine Failures**
   - **KittenTTS**: `kittentts package not installed` error
   - **Kokoro**: `'EspeakWrapper' has no attribute 'set_data_path'` error
   - **Solution**: Installed both engines with proper dependencies

3. **📚 Missing Dependencies**
   - **espeak-ng**: Required for Kokoro TTS phonemization
   - **misaki**: Required for Kokoro TTS text processing
   - **Solution**: Installed all required packages with correct versions

4. **📁 Import System Issues**
   - **Problem**: Relative import errors when running app directly
   - **Solution**: Added intelligent import handling for both module and direct execution

### **✅ What We Fixed:**

- **Python Environment**: Created new virtual environment with Python 3.12
- **KittenTTS**: Successfully installed and configured
- **Kokoro TTS**: Successfully installed with espeak-ng dependency
- **PDF Processing**: Completed full PDF to audiobook conversion functionality
- **UI Enhancement**: Added comprehensive PDF processing tab
- **Dependencies**: Resolved all package conflicts and version issues

---

## 🚀 **How to Run Next Time**

### **Prerequisites:**
- Python 3.12.x (not 3.13)
- macOS with Homebrew (for espeak-ng)

### **Step 1: Activate Virtual Environment**
```bash
cd /Users/anmol/Desktop/Code\ Projects/kitten_tts
source venv/bin/activate
```

### **Step 2: Verify Dependencies**
```bash
# Check Python version
python --version  # Should show Python 3.12.x

# Verify TTS engines
python -c "import kittentts; print('✅ KittenTTS working')"
python -c "import kokoro; print('✅ Kokoro working')"
```

### **Step 3: Launch Application**
```bash
python src/kitten_audiobook/ui/app.py
```

### **Step 4: Access Web Interface**
- **URL**: http://127.0.0.1:7861
- **Features Available**:
  - 📝 Text to Speech (both engines)
  - 📚 PDF to Audiobook conversion
  - 🎵 Audio playback and download
  - ⚙️ Voice and speed controls

---

## 🔧 **Troubleshooting**

### **If KittenTTS Fails:**
```bash
pip install --upgrade kittentts
```

### **If Kokoro Fails:**
```bash
# Reinstall espeak-ng
brew uninstall espeak-ng
brew install espeak-ng

# Reinstall Kokoro
pip install --force-reinstall "kokoro>=0.9.4"
```

### **If Import Errors Occur:**
```bash
# Ensure you're in the virtual environment
source venv/bin/activate

# Check Python path
python -c "import sys; print(sys.path)"
```

---

## 📦 **Package Versions (Working Configuration)**

```
Python: 3.12.11
KittenTTS: 0.1.0
Kokoro: 0.9.4
espeak-ng: 1.52.0
misaki: 0.9.4
espeakng-loader: 0.2.4
gradio: 5.42.0
torch: 2.8.0
```

---

## 🎯 **Key Features Working**

✅ **Dual TTS Engines**: Both KittenTTS and Kokoro fully functional  
✅ **PDF Processing**: Complete PDF to audiobook conversion  
✅ **Voice Selection**: Multiple voice options for each engine  
✅ **Audio Export**: Download generated audiobooks  
✅ **Progress Tracking**: Real-time conversion progress  
✅ **Error Handling**: Graceful fallbacks and user feedback  

---

## 📝 **Usage Instructions**

1. **Upload PDF**: Use the PDF tab to upload documents
2. **Select Engine**: Choose between KittenTTS or Kokoro
3. **Configure Voice**: Select voice and speed settings
4. **Convert**: Click "Create Audiobook" to start processing
5. **Download**: Get your generated audiobook file

---

## 🏆 **Success Status**

🎉 **MISSION ACCOMPLISHED**: All TTS engines working, PDF functionality complete, application fully operational!

**Last Tested**: Successfully running on http://127.0.0.1:7861 with both engines functional.
