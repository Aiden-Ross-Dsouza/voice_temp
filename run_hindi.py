#!/usr/bin/env python3
"""
Wrapper script to run voice_cloner.py with proper UTF-8 encoding.
"""

import subprocess
import sys

# Hindi text (properly encoded in this UTF-8 file)
hindi_text = "नमस्ते, यह एक परीक्षण है। मैं अपनी आवाज़ का उपयोग करके बोल रहा हूँ।"

# Build command
cmd = [
    sys.executable, "voice_cloner.py",
    "-r", "lady.wav",
    "-g", hindi_text,
    "-o", "hindi_output.wav",
    "--lang", "hi"
]

print(f"🎤 Running: {' '.join(cmd)}")
print(f"📝 Text: {hindi_text}")

# Run with UTF-8 encoding
result = subprocess.run(cmd, encoding='utf-8')
sys.exit(result.returncode)