# 🎤 Indic F5-TTS Voice Cloning

> **Zero-shot voice cloning for Indian languages** using F5-TTS (Faster, Smarter, Speech Synthesis with F5)
>
> Clone any voice in Hindi, Tamil, Kannada, Marathi, Telugu and more with high-quality synthesis.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [Voice Recording](#voice-recording)
  - [Voice Cloning](#voice-cloning)
  - [Processing Audio](#processing-audio)
- [Supported Languages](#supported-languages)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Configuration](#configuration)
- [Performance Tips](#performance-tips)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements **zero-shot voice cloning** for Indic languages using the F5-TTS framework. It allows you to:

- **Clone any voice** in mere seconds using reference audio
- **Support multiple Indic languages** (Hindi, Tamil, Kannada, Marathi, Telugu, etc.)
- **Generate natural-sounding speech** with minimal reference material
- **Preprocess audio** automatically with noise reduction and normalization
- **Record custom voices** directly from your microphone with built-in denoising

### Key Capabilities

| Feature | Details |
|---------|---------|
| **Languages** | Hindi, Tamil, Kannada, Marathi, Telugu, English |
| **Reference Audio** | 5-30 seconds minimum |
| **Voice Cloning Time** | <2 minutes per generation |
| **Audio Quality** | 24kHz sample rate, mono |
| **GPU Support** | CUDA 11.8+ recommended |
| **Installation** | Pip-based, single environment |

---

## ✨ Features

### Core Capabilities
- ✅ **Zero-shot voice cloning** - Clone voices with minimal reference audio
- ✅ **Multi-language support** - Works with 6+ Indic languages
- ✅ **Batch processing** - Clone multiple voices or generate multiple outputs
- ✅ **Audio preprocessing** - Automatic noise reduction and normalization
- ✅ **Voice recording** - Built-in microphone recording with denoising
- ✅ **Fine-tuned models** - Language-specific models for better quality

### Quality Enhancements
- 🎧 **Noise reduction** using spectral subtraction
- 🔊 **Volume normalization** with peak limiting
- ✂️ **Silence trimming** from audio edges
- 🎯 **Sample rate conversion** (auto-converted to 24kHz)

---

## 🔧 Requirements

### System Requirements
- **OS**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: NVIDIA GPU with CUDA support (highly recommended)
  - CUDA 11.8 or later
  - cuDNN 8.6+
  - Alternatively, use CPU mode (slower)

### Core Dependencies
See `requirements.txt` for the complete list. Key packages:
- `f5-tts>=1.1.18` - Core TTS engine
- `torch` and `torchaudio` - PyTorch deep learning framework
- `librosa` - Audio processing
- `sounddevice`, `soundfile` - Audio I/O
- `noisereduce` - Noise reduction

---

## 📦 Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/indic-voice-cloning.git
cd indic-voice-cloning
```

### Step 2: Create Virtual Environment
```bash
# Linux/macOS
python -m venv env
source env/bin/activate

# Windows
python -m venv env
env\Scripts\activate
```

### Step 3: Install PyTorch with CUDA

**For NVIDIA GPU (Recommended):**
```bash
# CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 (if you have newer GPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**
```bash
pip install torch torchaudio
```

**For Apple Silicon (M1/M2/M3):**
```bash
pip install torch torchaudio
```

### Step 4: Install Project Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Verify Installation
```bash
python -c "import f5_tts; import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🚀 Quick Start

### 5-Minute Voice Cloning

#### 1. **Prepare Reference Audio**
```bash
# Option A: Use existing audio file
# Ensure it's in WAV/MP3 format, 5-30 seconds, clear voice

# Option B: Record your own voice
python record_voice.py --output my_voice.wav --duration 15
```

#### 2. **Create Text Transcription**
Create a file `transcript.txt` with the exact text spoken in your audio:
```
Hello, this is a test of the voice cloning system.
I am speaking clearly so the system can learn my voice.
```

#### 3. **Clone the Voice**
```bash
python indic_f5_voice_clone.py \
  --reference_audio my_voice.wav \
  --reference_text transcript.txt \
  --output_text "Generate this new text with my voice" \
  --language hindi
```

#### 4. **Check Output**
Your cloned audio will be saved as `cloned_voice.wav` in the output directory.

---

## 📖 Usage Guide

### Voice Recording (`record_voice.py`)

Record high-quality reference audio directly from your microphone:

```bash
python record_voice.py [OPTIONS]
```

#### Options
| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `user_voice.wav` | Output file path |
| `--duration` | `-d` | `15` | Recording duration (seconds) |
| `--sample_rate` | `-sr` | `24000` | Sample rate in Hz |
| `--no_denoise` | - | False | Skip noise reduction |

#### Examples
```bash
# Record 20 seconds with noise reduction
python record_voice.py -o my_voice.wav -d 20

# Record without preprocessing (for testing)
python record_voice.py --no_denoise

# Use custom sample rate
python record_voice.py -sr 48000
```

#### Recording Tips
- ✅ Find a **quiet room** with minimal background noise
- ✅ Speak at **normal conversation volume**
- ✅ Pronounce words **clearly** (don't mumble)
- ✅ Avoid **sudden loud sounds** or rustling
- ✅ **Face the microphone** directly
- ✅ Keep **consistent speaking pace**

---

### Voice Cloning (`indic_f5_voice_clone.py`)

Generate speech in a cloned voice:

```bash
python indic_f5_voice_clone.py [OPTIONS]
```

#### Essential Arguments
```bash
--reference_audio FILE       Path to reference audio (WAV/MP3)
--reference_text FILE        Text file with reference audio transcription
--output_text TEXT           Text to synthesize in cloned voice
--language LANG              Target language (hindi, tamil, kannada, marathi, telugu)
```

#### Optional Arguments
```bash
--output_path PATH          Output directory (default: output/)
--output_name NAME          Output filename (default: cloned_voice.wav)
--seed SEED                 Random seed for reproducibility
--temperature TEMP          Voice variation (0.5-1.5, default: 1.0)
--top_p TOPP                Diversity control (0.5-1.0, default: 0.95)
--max_len LEN               Maximum output length in tokens
--chunk_size SIZE           Process in chunks (for long text)
--device DEVICE             cuda or cpu (auto-detect by default)
```

#### Examples

**Hindi Voice Cloning:**
```bash
python indic_f5_voice_clone.py \
  --reference_audio hindi_voice.wav \
  --reference_text hindi_transcript.txt \
  --output_text "नमस्ते, यह एक परीक्षण है" \
  --language hindi
```

**Tamil Voice Cloning:**
```bash
python indic_f5_voice_clone.py \
  --reference_audio tamil_voice.wav \
  --reference_text tamil_transcript.txt \
  --output_text "வணக்கம், இது ஒரு சோதனை" \
  --language tamil
```

**Batch Processing (Multiple Texts):**
```bash
# Create a file with multiple lines
echo "First text to clone" > texts.txt
echo "Second text to clone" >> texts.txt

# Process each line
while IFS= read -r text; do
  python indic_f5_voice_clone.py \
    --reference_audio reference.wav \
    --reference_text transcript.txt \
    --output_text "$text" \
    --language hindi
done < texts.txt
```

---

### Audio Preprocessing (`preprocess_audio.py`)

Manually preprocess audio files:

```bash
python preprocess_audio.py --input audio.wav --output clean_audio.wav
```

#### Features
- Automatic sample rate conversion to 24kHz
- Silence trimming
- Noise reduction
- Volume normalization

---

## 🌍 Supported Languages

| Language | Code | Status | Notes |
|----------|------|--------|-------|
| Hindi | `hindi` | ✅ Supported | Most tested |
| Tamil | `tamil` | ✅ Supported | Good quality |
| Kannada | `kannada` | ✅ Supported | Good quality |
| Marathi | `marathi` | ✅ Supported | Good quality |
| Telugu | `telugu` | ✅ Supported | Good quality |
| English | `english` | ✅ Supported | Works well |

### Language-Specific Tips

**Hindi**: Uses Devanagari script. Models are well-trained. Best results with native speakers.

**Tamil**: Uses Tamil script. Ensure proper Tamil text encoding (UTF-8).

**Kannada**: Less training data than Hindi. May need higher-quality reference audio.

**Marathi**: Similar to Hindi but with distinct phonetics. Clear pronunciation recommended.

**Telugu**: Uses Telugu script. Phonetic clarity is important.

---

## 📁 Project Structure

```
indic-voice-cloning/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── scripts/
│   ├── indic_f5_voice_clone.py       # Main voice cloning script
│   ├── record_voice.py               # Microphone recording utility
│   ├── preprocess_audio.py           # Audio preprocessing
│   ├── fix_torchaudio.py            # TorchAudio compatibility fixes
│   └── text.txt                      # Sample transcription
├── audio/                             # Input audio samples
│   ├── hindi_output.wav              # Sample Hindi output
│   ├── lady_hindi.wav                # Female Hindi reference
│   └── male.mp3                      # Male reference
├── output/                            # Generated cloned voices
│   ├── cloned_voice.wav              # Main output
│   ├── cloned_voice_lady_hindi.wav   # Lady voice sample
│   └── _ref_preprocessed.wav         # Preprocessed reference
├── voice_samples/                    # Reference voice samples
│   ├── hindi/long/
│   ├── tamil/long/
│   ├── kannada/long/
│   ├── marathi/long/
│   └── telugu/long/
└── notebooks/                         # Jupyter notebooks (examples)
    └── voice_cloning_demo.ipynb      # Interactive demo
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. **CUDA Not Found**
```
RuntimeError: CUDA is not available
```

**Solution:**
```bash
# Verify CUDA installation
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
```

#### 2. **Out of Memory (OOM)**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```bash
# Use CPU instead
python indic_f5_voice_clone.py ... --device cpu

# Reduce batch size or chunk size
python indic_f5_voice_clone.py ... --chunk_size 512

# Clear GPU cache
nvidia-smi | grep 'No running processes'
```

#### 3. **Model Download Issues**
```
ConnectionError: Failed to download model
```

**Solutions:**
```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/cache
python indic_f5_voice_clone.py ...

# Or use offline mode with pre-downloaded models
python indic_f5_voice_clone.py ... --offline_mode
```

#### 4. **Poor Audio Quality Output**
- ✅ Ensure reference audio is clear (SNR > 20dB)
- ✅ Use 15-30 seconds of reference audio
- ✅ Transcription must exactly match the audio
- ✅ Try with a different speaker if noise is persistent

#### 5. **Language Not Recognized**
```
ValueError: Unsupported language
```

**Solution:**
```bash
# Check available languages
python indic_f5_voice_clone.py --help

# Use supported language codes: hindi, tamil, kannada, marathi, telugu
```

#### 6. **Microphone Recording Issues**

**On Linux:**
```bash
# Install ALSA
sudo apt-get install libasound2-dev

# Test microphone
arecord -d 5 test.wav
```

**On macOS:**
```bash
# Check microphone permissions
System Preferences > Security & Privacy > Microphone
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# GPU settings
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0

# HuggingFace settings
export HF_HOME=/path/to/hf_cache
export HF_TOKEN=your_token_here

# Output settings
export OUTPUT_DIR=./output
export LOG_LEVEL=INFO
```

### Advanced Configuration

Edit the config parameters in scripts:

```python
# In indic_f5_voice_clone.py
CONFIG = {
    "sample_rate": 24000,           # Audio sample rate
    "max_duration": 300,            # Max output duration (seconds)
    "model_name": "f5-tts",         # Model variant
    "device": "cuda",               # Device (cuda/cpu)
    "dtype": "float32",             # Data type
    "seed": 42,                     # Random seed
}
```

---

## 🚄 Performance Tips

### For Faster Inference

1. **Use GPU:**
   ```bash
   python indic_f5_voice_clone.py ... --device cuda
   ```

2. **Use Smaller Models:**
   ```bash
   python indic_f5_voice_clone.py ... --model_size small
   ```

3. **Enable Mixed Precision:**
   ```bash
   export CUDA_LAUNCH_BLOCKING=0
   python indic_f5_voice_clone.py ... --mixed_precision
   ```

### For Better Quality

1. **Use Longer Reference Audio:**
   - 15-30 seconds is optimal
   - More data = better voice capture

2. **Increase Temperature Slightly:**
   ```bash
   python indic_f5_voice_clone.py ... --temperature 1.1
   ```

3. **Use Batch Processing:**
   ```bash
   # Process multiple texts together
   python indic_f5_voice_clone.py ... --batch_size 4
   ```

4. **Ensure Clean Reference:**
   ```bash
   python preprocess_audio.py --input noisy.wav --output clean.wav
   ```

---

## 📚 Examples

### Example 1: Clone a Movie Character's Voice (Hindi)

```bash
# 1. Extract reference audio clip
ffmpeg -i movie.mp4 -ss 00:05:20 -t 30 reference.wav

# 2. Transcribe the audio
# Manual: Open in Audacity, listen, and type exact words

# 3. Clone the voice
python indic_f5_voice_clone.py \
  --reference_audio reference.wav \
  --reference_text "यह मेरा पसंदीदा संवाद है" \
  --output_text "अब मैं कुछ नया कहूंगा" \
  --language hindi

# 4. Listen to output
# Check: output/cloned_voice.wav
```

### Example 2: Audiobook Narration (Tamil)

```bash
# 1. Record your voice
python record_voice.py -o narrator.wav -d 20

# 2. Transcribe recording
echo "I am your narrator for this book" > narrator_text.txt

# 3. Clone for multiple chapters
for chapter in chapter1.txt chapter2.txt chapter3.txt; do
  python indic_f5_voice_clone.py \
    --reference_audio narrator.wav \
    --reference_text narrator_text.txt \
    --output_text "$(cat $chapter)" \
    --language tamil \
    --output_name "narration_$(basename $chapter .txt).wav"
done
```

### Example 3: Accessibility Feature (Marathi)

```bash
# Convert written content to speech in user's voice

python indic_f5_voice_clone.py \
  --reference_audio user_voice.wav \
  --reference_text user_transcript.txt \
  --output_text "Important announcement: System maintenance tonight" \
  --language marathi \
  --output_path accessible_content/
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black scripts/
flake8 scripts/
```

### Areas for Contribution

- 🌍 Support for more Indic languages
- 🎨 Improved audio quality
- 📱 Web interface
- 📊 Performance optimization
- 📖 Documentation improvements
- 🧪 More test cases

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The F5-TTS model is based on research from multiple sources. Please cite appropriately if using this in academic work.

---

## 🙏 Acknowledgments

- **F5-TTS Team** for the core voice synthesis model
- **HuggingFace** for model hosting and tools
- **PyTorch** for deep learning framework
- **Community contributors** for language support and improvements

---

## 📞 Support & Contact

- 📧 **Issues**: Open a GitHub issue for bugs or questions
- 💬 **Discussions**: Use GitHub Discussions for general questions
- 🐦 **Twitter**: [@yourusername](https://twitter.com/)
- 📝 **Blog**: Check our blog for tutorials and updates

---

## 🔗 Quick Links

- [F5-TTS GitHub](https://github.com/f5-tts/f5-tts)
- [HuggingFace Models](https://huggingface.co/models?other=f5-tts)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [Indic Language Info](https://en.wikipedia.org/wiki/Indic_scripts)

---

## 📈 Roadmap

- [x] Multi-language support
- [x] Audio preprocessing
- [x] Voice recording utility
- [ ] Web UI dashboard
- [ ] Real-time streaming
- [ ] Voice conversion features
- [ ] Emotion control
- [ ] Multi-speaker support

---

**Last Updated**: May 2026  
**Version**: 2.0.0

Made with ❤️ for Indic languages
