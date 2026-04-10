#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F5-TTS Voice Cloning Script - FIXED VERSION
Clone any voice and generate new speech in that voice.
"""

import os
import sys
import random
import argparse
import soundfile as sf
from pathlib import Path

# ✅ ADD THIS: Import torch for device detection
import torch

# ✅ CORRECT IMPORT: F5TTS is in f5_tts.api
from f5_tts.api import F5TTS

# Optional: Whisper for auto-transcription
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


class VoiceCloner:
    """F5-TTS voice cloning with optional auto-transcription."""
    
    SUPPORTED_LANGUAGES = {
        'en': 'English', 'hi': 'Hindi', 'ta': 'Tamil', 'te': 'Telugu',
        'bn': 'Bengali', 'mr': 'Marathi', 'gu': 'Gujarati', 'kn': 'Kannada',
        'ml': 'Malayalam', 'pa': 'Punjabi', 'ur': 'Urdu', 'auto': 'Auto-detect'
    }
    
    def __init__(self, model="F5TTS_v1_Base", device=None, use_ema=True):
        """
        Initialize F5-TTS model.
        
        Args:
            model: "F5TTS_v1_Base" or "E2TTS_Base"
            device: "cuda", "cpu", "mps", or None for auto
            use_ema: Use EMA weights (recommended for quality)
        """
        print(f"🔄 Loading F5-TTS: {model}...")
        print("   💡 First run downloads ~2GB from HuggingFace")
        
        # Auto-detect device if not specified
        self.device = device
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"   🎮 Using CUDA: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
                print("   🍎 Using Apple MPS")
            else:
                self.device = "cpu"
                print("   💻 Using CPU (will be slower)")
        
        self.f5tts = F5TTS(
            model=model,
            device=self.device,
            use_ema=use_ema,
            ode_method="euler",
        )
        print(f"✅ Model loaded on {self.f5tts.device}")
        
        # Load Whisper if available and requested
        self.whisper_model = None
    
    def load_whisper(self, device=None):
        """Load Whisper model for auto-transcription."""
        if not WHISPER_AVAILABLE:
            print("⚠️  Whisper not installed. Install: pip install openai-whisper")
            return False
        
        try:
            dev = device or self.device
            print(f"🔄 Loading Whisper ({dev})...")
            self.whisper_model = whisper.load_model("medium", device=dev)
            print("✅ Whisper ready")
            return True
        except Exception as e:
            print(f"⚠️  Whisper load failed: {e}")
            return False
    
    def transcribe(self, audio_path, language=None):
        """Auto-transcribe reference audio."""
        if not self.whisper_model:
            if not self.load_whisper():
                raise RuntimeError("Whisper not available. Provide ref_text manually.")
        
        print(f"🎤 Transcribing: {os.path.basename(audio_path)}...")
        result = self.whisper_model.transcribe(audio_path, language=language)
        text = result["text"].strip()
        print(f"📝 '{text}'")
        return text
    
    def clone(self, ref_audio, ref_text, gen_text, output_path,
              remove_silence=True, cfg_strength=2.0, nfe_step=32,
              speed=1.0, seed=None, lang=None):
        """
        Generate speech in cloned voice.
        
        Args:
            ref_audio: Reference audio file path
            ref_text: Text spoken in reference (empty = auto-transcribe)
            gen_text: Target text to generate
            output_path: Output WAV file path
            remove_silence: Trim output silence
            cfg_strength: CFG strength (1.5-3.0)
            nfe_step: ODE steps (16-64, higher = better/slower)
            speed: Speed multiplier (0.8-1.2)
            seed: Random seed for reproducibility
            lang: Language code for Whisper ('hi', 'ta', etc.)
        """
        # Handle reference text
        if not ref_text.strip():
            if lang == 'auto':
                lang = None  # Whisper auto-detect
            ref_text = self.transcribe(ref_audio, language=lang)
        
        # Set seed
        if seed is None:
            seed = random.randint(0, 2**31)
        
        print(f"\n🎯 Generating speech...")
        print(f"   Reference: {os.path.basename(ref_audio)}")
        print(f"   Ref text: '{ref_text[:60]}{'...' if len(ref_text)>60 else ''}'")
        print(f"   Target: '{gen_text[:60]}{'...' if len(gen_text)>60 else ''}'")
        print(f"   Output: {output_path}")
        print(f"   Settings: CFG={cfg_strength}, NFE={nfe_step}, Speed={speed}x")
        
        try:
            # ✅ F5-TTS infer() returns (wav, sr, spec)
            wav, sr, spec = self.f5tts.infer(
                ref_file=ref_audio,
                ref_text=ref_text,
                gen_text=gen_text,
                file_wave=output_path,  # Auto-saves
                remove_silence=remove_silence,
                cfg_strength=cfg_strength,
                nfe_step=nfe_step,
                speed=speed,
                seed=seed,
                show_info=print,
            )
            
            duration = len(wav) / sr
            print(f"\n✅ Generated: {output_path}")
            print(f"   Duration: {duration:.2f}s | SR: {sr}Hz | Seed: {seed}")
            return output_path
            
        except Exception as e:
            print(f"\n❌ Generation failed: {e}")
            raise


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="F5-TTS Voice Cloning - Zero-shot TTS with voice cloning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Language codes for --lang: {', '.join(VoiceCloner.SUPPORTED_LANGUAGES.keys())}

Examples:
  # English with manual reference text
  python voice_cloner.py -r ref.wav -t "Hello world" -g "New speech here" -o out.wav

  # Hindi with auto-transcription
  python voice_cloner.py -r hindi.wav -g "नमस्ते, यह परीक्षण है" -o out.wav --lang hi

  # Higher quality (slower)
  python voice_cloner.py -r ref.wav -t "ref" -g "target" -o out.wav --nfe 64 --cfg 2.5

  # CPU-only (slow but works without GPU)
  python voice_cloner.py -r ref.wav -g "text" -o out.wav --device cpu

  # Reproducible output
  python voice_cloner.py -r ref.wav -t "ref" -g "target" -o out.wav --seed 42
        """
    )
    
    # Required args
    parser.add_argument('-r', '--ref-audio', required=True, help='Reference audio file')
    parser.add_argument('-g', '--gen-text', required=True, help='Text to generate')
    
    # Optional args
    parser.add_argument('-t', '--ref-text', default='', help='Reference text (auto if empty)')
    parser.add_argument('-o', '--output', default='output.wav', help='Output file path')
    parser.add_argument('--lang', default=None, choices=list(VoiceCloner.SUPPORTED_LANGUAGES.keys()),
                        help='Language code for Whisper transcription')
    
    # Model settings
    parser.add_argument('--model', default='F5TTS_v1_Base', 
                        choices=['F5TTS_v1_Base', 'E2TTS_Base'],
                        help='Model variant (default: F5TTS_v1_Base)')
    parser.add_argument('--device', default=None, choices=['cuda', 'cpu', 'mps'],
                        help='Device (auto-detect if not specified)')
    parser.add_argument('--no-ema', action='store_true', help='Disable EMA weights')
    
    # Generation settings
    parser.add_argument('--nfe', type=int, default=32, choices=[16, 24, 32, 48, 64],
                        help='ODE steps: 16=fast, 64=best quality (default: 32)')
    parser.add_argument('--cfg', type=float, default=2.0, 
                        help='CFG strength 1.5-3.0 (default: 2.0)')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Speed multiplier 0.8-1.2 (default: 1.0)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--no-silence', action='store_true', help='Keep output silence')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.ref_audio):
        print(f"❌ Reference audio not found: {args.ref_audio}")
        return 1
    
    # Initialize
    try:
        cloner = VoiceCloner(
            model=args.model,
            device=args.device,
            use_ema=not args.no_ema
        )
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("\n💡 Troubleshooting:")
        print("   • Install PyTorch with CUDA: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("   • First run downloads model - check internet")
        print("   • For CPU: add --device cpu (will be slow)")
        return 1
    
    # Generate
    try:
        cloner.clone(
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            gen_text=args.gen_text,
            output_path=args.output,
            remove_silence=not args.no_silence,
            cfg_strength=args.cfg,
            nfe_step=args.nfe,
            speed=args.speed,
            seed=args.seed,
            lang=args.lang,
        )
        print(f"\n🎉 Success! Output: {args.output}")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())