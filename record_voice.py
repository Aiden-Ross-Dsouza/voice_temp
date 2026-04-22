import argparse
import sys
import queue
import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
    import noisereduce as nr
    import librosa
except ImportError:
    print("Missing dependencies! Please run:")
    print("pip install sounddevice soundfile noisereduce librosa numpy")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Record browser/microphone audio, denoise it, and save for Voice Cloning.")
    parser.add_argument("--output", "-o", type=str, default="user_voice.wav", help="Output file path (.wav)")
    parser.add_argument("--duration", "-d", type=int, default=15, help="Recording duration in seconds")
    parser.add_argument("--sample_rate", "-sr", type=int, default=24000, help="Sample rate (24000 recommended for IndicF5)")
    parser.add_argument("--no_denoise", action="store_true", help="Skip noise reduction")
    args = parser.parse_args()

    print("\n🎤 Voice Recording Utility 🎤")
    print("═════════════════════════════\n")
    
    # 1. Select Device
    print("Available devices:")
    print(sd.query_devices())
    device_id = None
    try:
        user_input = input("\nEnter the device ID to use (or press Enter for default microphone): ")
        if user_input.strip():
            device_id = int(user_input)
    except ValueError:
        print("Invalid ID. Using default microphone.")

    print(f"\n🎙️  Get ready! Recording for {args.duration} seconds...")
    print("Speak clearly as if reading a story. (Wait for 'GO!'...)")
    
    # Optional wait loop to prepare themselves
    import time
    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)
    print("🚀 GO! Recording now...")

    # 2. Record audio
    try:
        recording = sd.rec(
            int(args.duration * args.sample_rate),
            samplerate=args.sample_rate,
            channels=1,
            dtype='float32',
            device=device_id
        )
        for i in range(args.duration):
            time.sleep(1)
            print(f"  ... {args.duration - i - 1} seconds left")
        sd.wait()
    except Exception as e:
        print(f"\n❌ Error recording audio: {e}")
        sys.exit(1)

    print("\n✅ Recording finished!")

    # Format the data
    audio_data = np.squeeze(recording)

    # 3. Trim silence
    print("✂️  Trimming silence from start/end...")
    audio_data, _ = librosa.effects.trim(audio_data, top_db=30)

    # 4. Noise Reduction
    if not args.no_denoise:
        print("🧹 Running Noise Reduction (Spectral Subtraction)...")
        # Assume first 0.5s is background noise for profiling
        noise_profile_len = min(int(args.sample_rate * 0.5), len(audio_data))
        if noise_profile_len > 0:
            audio_data = nr.reduce_noise(
                y=audio_data, 
                sr=args.sample_rate, 
                prop_decrease=0.8,
                stationary=True
            )

    # 5. Normalize Audio
    print("🔊 Normalizing audio volume...")
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val
        # Peak normalize to -1.0 dB to prevent clipping
        audio_data = audio_data * (10 ** (-1.0 / 20))

    # 6. Save
    sf.write(args.output, audio_data, args.sample_rate)
    duration_actual = len(audio_data) / args.sample_rate
    
    print("\n═════════════════════════════")
    print(f"🎉 Success! Audio saved to: '{args.output}'")
    print(f"Final Duration: {duration_actual:.2f} seconds")
    print(f"Sample Rate: {args.sample_rate} Hz (IndicF5 ready)")
    print("\n📝 Next Step: Transcribe this audio exactly into a .txt file, then run your cloning script!")

if __name__ == "__main__":
    main()
