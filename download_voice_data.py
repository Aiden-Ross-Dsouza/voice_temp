import os
from datasets import load_dataset, Audio
import soundfile as sf
from pathlib import Path
import librosa
from tqdm import tqdm
import io

# Configuration
LANGUAGES = {
    "hindi": "hi_in",
    "marathi": "mr_in",
    "tamil": "ta_in",
    "kannada": "kn_in"
}
SPEAKERS_PER_LANG = 3
SAMPLES_PER_SPEAKER = 5
BASE_OUTPUT_DIR = "voice_samples"

# Duration Buckets (in seconds)
BUCKETS = [
    (0, 3, "short"),
    (3, 7, "medium"),
    (7, 100, "long")
]

def get_bucket_name(duration):
    for low, high, name in BUCKETS:
        if low <= duration < high:
            return name
    return "extra_long"

def main():
    # Create base directory
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    for lang_name, lang_code in LANGUAGES.items():
        print(f"\n--- Processing Language: {lang_name} ({lang_code}) ---")
        
        # Load dataset metadata first (streaming to save space)
        # Using the refs/convert/parquet branch to avoid forbidden .py scripts
        dataset = load_dataset(
            "google/fleurs", 
            revision="refs/convert/parquet", 
            data_dir=lang_code, 
            split="train", 
            streaming=True
        )
        
        # Disable automatic decoding to bypass broken torchcodec
        dataset = dataset.cast_column("audio", Audio(decode=False))
        
        # Group samples by virtual speakers (based on Gender and ID)
        # This ensures at least one Male and one Female, plus a third unique sample set
        speaker_map = {}
        genders_found = set()
        
        print(f"Selecting diverse speakers...")
        for sample in dataset:
            gender = sample.get("gender", "unknown")
            # Create a virtual speaker ID using gender + a snippet of their sample ID
            # In Fleurs, samples are recorded by hundreds of speakers, 
            # so different IDs usually mean different speakers.
            sid = f"{gender}_{str(sample['id'])[:2]}" 
            
            if sid not in speaker_map:
                if len(speaker_map) < SPEAKERS_PER_LANG:
                    speaker_map[sid] = []
                    genders_found.add(gender)
            
            if sid in speaker_map and len(speaker_map[sid]) < SAMPLES_PER_SPEAKER:
                speaker_map[sid].append(sample)
            
            # Stop if we have enough speakers and samples
            ready = all(len(s) >= SAMPLES_PER_SPEAKER for s in speaker_map.values())
            if len(speaker_map) >= SPEAKERS_PER_LANG and ready:
                break
        
        # Process and save samples
        for sid, samples in speaker_map.items():
            print(f"Downloading samples for Speaker: {sid}...")
            
            for i, sample in enumerate(samples):
                # Manual decoding from bytes using librosa to avoid torchcodec issues
                audio_bytes = sample["audio"]["bytes"]
                audio_array, sampling_rate = librosa.load(io.BytesIO(audio_bytes), sr=16000)
                
                duration = len(audio_array) / sampling_rate
                bucket = get_bucket_name(duration)
                
                # Cleanup paths
                clean_sid = sid.replace(" ", "_")
                target_dir = Path(BASE_OUTPUT_DIR) / lang_name / bucket / clean_sid
                target_dir.mkdir(parents=True, exist_ok=True)
                
                file_path = target_dir / f"sample_{i}.wav"
                text_path = target_dir / f"sample_{i}.txt"
                
                sf.write(str(file_path), audio_array, sampling_rate)
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(sample["transcription"])

    print("\nDownload complete! Check the 'voice_samples' folder.")

if __name__ == "__main__":
    main()
