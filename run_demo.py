#!/usr/bin/env python3
"""
Quick demo script for F5-TTS voice cloning.
Creates sample output with minimal setup.

Usage:
    python run_demo.py
"""

import os
import sys
import tempfile
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from preprocess_audio import AudioPreprocessor
from voice_cloner import VoiceCloner


def create_sample_reference(output_path="demo_ref.wav"):
    """Create a simple reference audio for testing (requires scipy)."""
    import numpy as np
    import soundfile as sf
    
    print("🎵 Creating sample reference audio...")
    
    # Simple sine wave with speech-like modulation
    sr = 22050
    duration = 3  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Base frequency + modulation to simulate speech
    freq = 200 + 50 * np.sin(2 * np.pi * 2 * t)  # Varying pitch
    audio = 0.3 * np.sin(2 * np.pi * freq * t)
    
    # Add simple "envelope" to mimic speech rhythm
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    audio = audio * envelope
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    sf.write(output_path, audio, sr, subtype='PCM_16')
    print(f"✅ Created: {output_path}")
    return output_path


def demo():
    """Run a complete demo."""
    print("\n" + "=" * 60)
    print("🚀 F5-TTS Voice Cloning Demo")
    print("=" * 60 + "\n")
    
    # Step 1: Create or find reference audio
    ref_audio = "demo_ref.wav"
    ref_text = "This is a sample reference for voice cloning."
    
    if not os.path.exists(ref_audio):
        print("📦 No reference audio found. Creating sample...")
        ref_audio = create_sample_reference(ref_audio)
    
    # Step 2: Preprocess (optional but recommended)
    print("\n🔧 Preprocessing reference audio...")
    try:
        preprocessor = AudioPreprocessor()
        ref_processed = preprocessor.preprocess(
            input_path=ref_audio,
            output_path="demo_ref_processed.wav",
            trim_silence=True,
            normalize=True,
            max_duration=15
        )
        ref_audio = ref_processed
    except Exception as e:
        print(f"⚠️  Preprocessing skipped: {e}")
    
    # Step 3: Initialize cloner
    print("\n🤖 Loading F5-TTS model...")
    try:
        cloner = VoiceCloner(model="F5TTS_v1_Base", device=None)
    except Exception as e:
        print(f"❌ Failed: {e}")
        print("\n💡 Make sure you've installed dependencies:")
        print("   pip install -r requirements.txt")
        return 1
    
    # Step 4: Generate speech
    target_texts = [
        "Hello! This is cloned speech using F5-TTS.",
        "Voice cloning technology enables amazing applications.",
        "Always use this technology responsibly and ethically.",
    ]
    
    print(f"\n🎤 Generating {len(target_texts)} samples...")
    outputs = []
    
    for i, text in enumerate(target_texts, 1):
        output_path = f"demo_output_{i}.wav"
        try:
            cloner.clone(
                ref_audio=ref_audio,
                ref_text=ref_text,
                gen_text=text,
                output_path=output_path,
                nfe_step=24,  # Faster for demo
                cfg_strength=2.0,
                seed=42 + i,  # Different seed per sample
            )
            outputs.append(output_path)
        except Exception as e:
            print(f"⚠️  Sample {i} failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Demo Summary")
    print("=" * 60)
    print(f"Reference: {ref_audio}")
    print(f"Generated: {len(outputs)}/{len(target_texts)} samples")
    for out in outputs:
        if os.path.exists(out):
            size = os.path.getsize(out) / 1024
            print(f"   ✅ {out} ({size:.1f} KB)")
    print("=" * 60)
    
    if outputs:
        print("\n🎧 Play with: python -m soundfile play demo_output_1.wav")
        print("   Or open in any audio player (VLC, Audacity, etc.)")
    
    return 0


if __name__ == "__main__":
    sys.exit(demo())