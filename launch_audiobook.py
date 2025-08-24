#!/usr/bin/env python3
"""
Launch script for the Kitten TTS Audiobook Generator
"""

import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)

def main():
    """Launch the Kitten TTS Audiobook application"""
    try:
        print("🎙️ Starting Kitten TTS Audiobook Generator...")
        print("📚 Features available:")
        print("  - Text-to-Speech conversion")
        print("  - PDF to Audiobook conversion")
        print("  - Multiple voice options")
        print("  - Real-time word highlighting")
        print("  - Progress tracking")
        print("")
        
        from src.kitten_audiobook.ui.app import build_ui
        
        # Build and launch the interface
        demo = build_ui()
        
        print("🚀 Launching web interface...")
        print("📱 Open your browser to the URL that appears below")
        print("📄 You can test with the included 'test_document.pdf'")
        print("")
        
        # Launch with public sharing disabled for security
        demo.launch(
            server_name="127.0.0.1",  # Local only
            server_port=7860,         # Default Gradio port
            share=False,              # No public sharing
            show_tips=False,          # Cleaner interface
            show_error=True,          # Show errors for debugging
            quiet=False               # Show startup messages
        )
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye! Application stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("💡 Make sure you're in the correct directory and virtual environment is activated")
        sys.exit(1)

if __name__ == "__main__":
    main()