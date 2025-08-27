#!/usr/bin/env python3

try:
    from kokoro import KPipeline
    print('✅ Kokoro successfully imported')
    
    pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    print('✅ Kokoro pipeline created successfully')
    
    # Test voice availability
    available_voices = ['af_alloy', 'af_heart', 'af_nova', 'am_adam', 'am_echo']
    print(f'🎤 Testing voices: {available_voices[:3]}...')
    
except Exception as e:
    print(f'❌ Kokoro error: {e}')
    print('🔧 Kokoro is not properly installed')