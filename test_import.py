#!/usr/bin/env python3
"""
Quick verification that F5-TTS is installed correctly.
Run this first before using the main scripts.
"""

import sys
import torch

def check_torch():
    """Verify PyTorch installation."""
    print("🔍 Checking PyTorch...")
    print(f"   Version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
    return True

def check_f5_tts():
    """Verify F5-TTS installation and import."""
    print("\n🔍 Checking F5-TTS...")
    try:
        # ✅ CORRECT IMPORT PATH
        from f5_tts.api import F5TTS
        print("   ✅ Import successful: f5_tts.api.F5TTS")
        return True
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        print("\n💡 Try reinstalling:")
        print("   pip uninstall f5-tts")
        print("   pip install f5-tts --no-cache-dir")
        return False

def check_optional():
    """Check optional dependencies."""
    print("\n🔍 Checking optional dependencies...")
    
    # Whisper
    try:
        import whisper
        print("   ✅ Whisper: available")
    except ImportError:
        print("   ⚠️  Whisper: not installed (auto-transcription disabled)")
    
    # Librosa
    try:
        import librosa
        print("   ✅ Librosa: available")
    except ImportError:
        print("   ❌ Librosa: not installed (preprocessing disabled)")
    
    # SoundFile
    try:
        import soundfile
        print("   ✅ SoundFile: available")
    except ImportError:
        print("   ❌ SoundFile: not installed")

def main():
    print("=" * 60)
    print("F5-TTS Installation Verification")
    print("=" * 60)
    
    checks = [
        check_torch(),
        check_f5_tts(),
    ]
    check_optional()
    
    print("\n" + "=" * 60)
    if all(checks):
        print("✅ All critical checks passed! Ready to clone voices.")
        print("\n🚀 Next steps:")
        print("   1. Run: python preprocess_audio.py -i your_audio.wav")
        print("   2. Run: python voice_cloner.py -r processed.wav -g 'Your text'")
        return 0
    else:
        print("❌ Some checks failed. Please fix issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())