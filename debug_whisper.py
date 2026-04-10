#!/usr/bin/env python3
"""Debug what Whisper hears from your Hindi audio."""

import whisper
import sys

audio_file = sys.argv[1] if len(sys.argv) > 1 else "lady.wav"

print(f" Analyzing: {audio_file}")
print("=" * 60)

# Try different models
models = ["base", "small", "medium"]

for model_name in models:
    print(f"\n📊 Model: {model_name}")
    print("-" * 60)
    
    try:
        model = whisper.load_model(model_name)
        
        # Try with Hindi
        result_hi = model.transcribe(audio_file, language="hi")
        print(f"  Hindi (hi): {result_hi['text']}")
        
        # Try auto-detect
        result_auto = model.transcribe(audio_file)
        print(f"  Auto:       {result_auto['text']}")
        print(f"  Detected:   {result_auto['language']}")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "=" * 60)
print("💡 If Hindi output shows Arabic script, use -t flag for manual text")