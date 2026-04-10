# 🎤 F5-TTS Voice Cloning Project

Zero-shot voice cloning using F5-TTS with support for Indic languages.

## ⚡ Quick Start

### 1. Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv env
source env/bin/activate  # Linux/Mac
# or: env\Scripts\activate  # Windows

# Install PyTorch with CUDA (adjust for your system)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
pip install -r requirements.txt

# Verify installation
python test_import.py