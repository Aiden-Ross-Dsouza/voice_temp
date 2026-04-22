#!/usr/bin/env python3
"""
Audio Preprocessor for F5-TTS Voice Cloning
Converts any audio format to F5-TTS compatible WAV format.

Usage:
    python preprocess_audio.py -i input.mp3 -o output.wav
    python preprocess_audio.py -i recording.wav --noise-reduction
"""

import os
import sys
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️  librosa not installed. Install with: pip install librosa")


class AudioPreprocessor:
    """Preprocesses audio files for F5-TTS voice cloning."""
    
    # F5-TTS optimal specifications
    TARGET_SR = 22050
    TARGET_CHANNELS = 1
    MIN_DURATION = 8      # seconds
    MAX_DURATION = 30     # seconds
    TARGET_DB = -20       # normalization level
    
    def __init__(self, target_sr=None):
        if target_sr:
            self.TARGET_SR = target_sr
    
    def load_audio(self, input_path):
        """Load audio from any supported format."""
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa required for audio loading")
        
        print(f"📥 Loading: {input_path}")
        audio, sr = librosa.load(input_path, sr=None, mono=False)
        
        info = {
            'sr': sr,
            'channels': audio.shape[0] if len(audio.shape) > 1 else 1,
            'duration': len(audio) / sr if audio.ndim == 1 else len(audio[0]) / sr,
            'samples': len(audio) if audio.ndim == 1 else len(audio[0])
        }
        print(f"   📊 {info['sr']}Hz | {info['channels']}ch | {info['duration']:.2f}s")
        return audio, sr, info
    
    def to_mono(self, audio):
        """Convert to mono if stereo."""
        if audio.ndim > 1:
            print("🔀 Converting to mono...")
            return librosa.to_mono(audio)
        return audio
    
    def resample(self, audio, orig_sr, target_sr):
        """Resample to target sample rate."""
        if orig_sr != target_sr:
            print(f"🔄 Resampling: {orig_sr}Hz → {target_sr}Hz")
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        return audio
    
    def trim_silence(self, audio, top_db=20):
        """Remove silence from edges."""
        print("✂️  Trimming silence...")
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed
    
    def normalize(self, audio, target_db=None):
        """Normalize audio to target RMS level."""
        if target_db is None:
            target_db = self.TARGET_DB
        
        print(f"🔊 Normalizing to {target_db}dB...")
        
        # Calculate current RMS in dB
        rms = np.sqrt(np.mean(audio ** 2))
        current_db = 20 * np.log10(rms + 1e-8)
        
        # Apply gain
        gain_db = target_db - current_db
        gain_linear = 10 ** (gain_db / 20)
        normalized = audio * gain_linear
        
        # Prevent clipping
        peak = np.max(np.abs(normalized))
        if peak > 0.99:
            normalized = normalized * (0.99 / peak)
            print("   ⚠️  Applied clipping prevention")
        
        return normalized
    
    def trim_duration(self, audio, sr, max_dur=None, min_dur=None):
        """Ensure audio is within duration bounds."""
        if max_dur is None:
            max_dur = self.MAX_DURATION
        if min_dur is None:
            min_dur = self.MIN_DURATION
        
        duration = len(audio) / sr
        print(f"⏱️  Duration: {duration:.2f}s")
        
        if duration > max_dur:
            print(f"✂️  Trimming to {max_dur}s (taking middle portion)...")
            target_samples = int(max_dur * sr)
            start = (len(audio) - target_samples) // 2
            return audio[start:start + target_samples]
        
        elif duration < min_dur:
            print(f"⚠️  Warning: Audio shorter than recommended {min_dur}s")
            print("   💡 Longer reference audio (15-25s) gives better cloning")
        
        return audio
    
    def simple_noise_reduce(self, audio, strength=1.0):
        """Advanced noise reduction using noisereduce (or basic fallback)."""
        
        try:
            import noisereduce as nr
            print(f"🧹 Applying advanced noise reduction (aggression: {strength})...")
            # Using stationary=False is often more aggressive for complex backgrounds
            return nr.reduce_noise(
                y=audio, 
                sr=self.TARGET_SR, 
                prop_decrease=strength, 
                stationary=False 
            )
        except ImportError:
            print("   ⚠️  'noisereduce' package not found. Falling back to basic STFT method...")
            print("🧹 Applying basic noise reduction...")
            # STFT
            D = librosa.stft(audio)
            magnitude = np.abs(D)
            phase = np.angle(D)
            
            # Estimate noise floor
            noise_floor = np.percentile(magnitude, 10, axis=1, keepdims=True)
            
            # Spectral gating
            magnitude_clean = np.maximum(magnitude - strength * noise_floor, 0)
            
            # Reconstruct
            D_clean = magnitude_clean * np.exp(1j * phase)
            cleaned = librosa.istft(D_clean, length=len(audio))
            return cleaned
    
    def preprocess(self, input_path, output_path=None, 
                   trim_silence=True, normalize=True,
                   noise_reduce=True, max_duration=30):
        """Complete preprocessing pipeline."""
        print("\n" + "=" * 60)
        print("🎵 F5-TTS Audio Preprocessor")
        print("=" * 60)
        
        if not LIBROSA_AVAILABLE:
            raise RuntimeError("librosa is required. Install with: pip install librosa")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Generate output path
        if output_path is None:
            stem = Path(input_path).stem
            output_path = f"{stem}_processed.wav"
        
        # Load
        audio, sr, info = self.load_audio(input_path)
        
        # Process pipeline
        audio = self.to_mono(audio)
        audio = self.resample(audio, sr, self.TARGET_SR)
        
        if trim_silence:
            audio = self.trim_silence(audio)
        
        if noise_reduce:
            audio = self.simple_noise_reduce(audio)
        
        if normalize:
            audio = self.normalize(audio)
        
        audio = self.trim_duration(audio, self.TARGET_SR, max_duration)
        
        # Save
        print(f"💾 Saving: {output_path}")
        sf.write(output_path, audio, self.TARGET_SR, subtype='PCM_16')
        
        # Verify
        verify, verify_sr = sf.read(output_path)
        file_size = os.path.getsize(output_path) / 1024
        
        print("\n" + "=" * 60)
        print("✅ Preprocessing Complete!")
        print("=" * 60)
        print(f"   Output: {output_path}")
        print(f"   Format: WAV, 16-bit PCM, Mono")
        print(f"   Sample Rate: {verify_sr} Hz")
        print(f"   Duration: {len(verify)/verify_sr:.2f} seconds")
        print(f"   File Size: {file_size:.1f} KB")
        print("=" * 60 + "\n")
        
        return output_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess audio for F5-TTS voice cloning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python preprocess_audio.py -i voice.mp3
  python preprocess_audio.py -i recording.wav -o reference.wav
  python preprocess_audio.py -i audio.m4a --noise-reduction --max-duration 20
  python preprocess_audio.py -i input.wav --no-normalize
        """
    )
    
    parser.add_argument('-i', '--input', required=True, help='Input audio file')
    parser.add_argument('-o', '--output', default=None, help='Output file path')
    parser.add_argument('--sample-rate', type=int, default=22050, 
                        help='Target sample rate (default: 22050)')
    parser.add_argument('--no-trim-silence', action='store_true', 
                        help='Disable silence trimming')
    parser.add_argument('--no-normalize', action='store_true', 
                        help='Disable normalization')
    parser.add_argument('--no-noise-reduction', action='store_true', 
                        help='Disable noise reduction')
    parser.add_argument('--max-duration', type=int, default=30, 
                        help='Max duration in seconds (default: 30)')
    
    args = parser.parse_args()
    
    try:
        preprocessor = AudioPreprocessor(target_sr=args.sample_rate)
        output = preprocessor.preprocess(
            input_path=args.input,
            output_path=args.output,
            trim_silence=not args.no_trim_silence,
            normalize=not args.no_normalize,
            noise_reduce=not args.no_noise_reduction,
            max_duration=args.max_duration
        )
        print(f"🎯 Ready for F5-TTS: {output}")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())