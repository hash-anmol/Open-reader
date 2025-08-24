#!/usr/bin/env python3
"""
Test script for the integrated TTS functionality
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_kitten_tts():
    """Test KittenTTS functionality"""
    try:
        from kitten_audiobook.tts.engine import TTSEngine, TTSConfig
        print("✅ KittenTTS engine imports successfully")
        
        engine = TTSEngine(TTSConfig())
        print("✅ KittenTTS engine initialized")
        
        # Test synthesis
        test_text = "Hello, this is a test of KittenTTS."
        audio = engine.synthesize(test_text, voice="expr-voice-4-f", speed=1.0)
        
        if audio is not None and len(audio) > 0:
            print(f"✅ KittenTTS synthesis successful - generated {len(audio)} samples")
            return True
        else:
            print("❌ KittenTTS synthesis failed - no audio generated")
            return False
            
    except Exception as e:
        print(f"❌ KittenTTS test failed: {e}")
        return False

def test_kokoro_tts():
    """Test Kokoro TTS functionality"""
    try:
        from kokoro import KPipeline
        print("✅ Kokoro imports successfully")
        
        pipeline = KPipeline(lang_code='a')
        print("✅ Kokoro pipeline initialized")
        
        # Test synthesis
        test_text = "Hello, this is a test of Kokoro TTS."
        segments = pipeline(test_text, voice="af_heart", speed=1.0)
        
        audio_parts = []
        for segment in segments:
            if hasattr(segment, 'audio'):
                audio_data = segment.audio
                if hasattr(audio_data, 'detach'):
                    audio_data = audio_data.detach().cpu().numpy()
                if len(audio_data) > 0:
                    audio_parts.append(audio_data)
        
        if audio_parts:
            total_samples = sum(len(part) for part in audio_parts)
            print(f"✅ Kokoro synthesis successful - generated {total_samples} samples")
            return True
        else:
            print("❌ Kokoro synthesis failed - no audio generated")
            return False
            
    except Exception as e:
        print(f"❌ Kokoro test failed: {e}")
        return False

def main():
    print("🧪 Testing TTS Integration")
    print("=" * 50)
    
    kitten_works = test_kitten_tts()
    print()
    kokoro_works = test_kokoro_tts()
    
    print()
    print("📊 Test Results:")
    print(f"   • KittenTTS: {'✅ Working' if kitten_works else '❌ Failed'}")
    print(f"   • Kokoro-82M: {'✅ Working' if kokoro_works else '❌ Failed'}")
    
    if kitten_works or kokoro_works:
        print("\n🎉 At least one TTS engine is working!")
        print("🚀 Your enhanced interface should have real TTS synthesis available.")
    else:
        print("\n⚠️ No TTS engines are working.")
        print("🔧 The interface will run in demo mode with mock audio.")
    
    print(f"\n🌐 Access your enhanced interface at: http://127.0.0.1:7862")

if __name__ == "__main__":
    main()