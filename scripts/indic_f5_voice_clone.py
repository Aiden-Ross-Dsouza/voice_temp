"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         IndicF5 End-to-End Voice Cloning Pipeline                           ║
║         AI4Bharat IndicF5 — 11 Indian Languages, Near-Human TTS             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Supported Languages:
    Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam,
    Marathi, Odia, Punjabi, Tamil, Telugu

Pipeline Steps:
    1. Load & validate reference audio
    2. Preprocess audio (denoise → resample → trim silence → normalize → clip to 15s)
    3. Auto-transcribe reference audio (Whisper / AI4Bharat ASR)
    4. Load IndicF5 model
    5. Synthesize speech in cloned voice
    6. Post-process & save output
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

# ─── Windows UTF-8 Fix (MUST be before any other I/O) ────────────────────────
if sys.platform == "win32":
    import io
    # Force stdout/stderr to UTF-8 so Indic scripts print correctly
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    # Also set the environment variable for child processes
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def _setup_local_cache(base_dir: str | None = None) -> str:
    # Redirect ALL model caches to a local folder off the C: drive.
    if base_dir is None:
        # Place cache/ right next to this script so everything stays together
        base_dir = str(Path(__file__).resolve().parent / "cache")

    cache_root = Path(base_dir).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    subdirs = {
        "HF_HOME":               cache_root / "huggingface",
        "HUGGINGFACE_HUB_CACHE": cache_root / "huggingface" / "hub",
        "TRANSFORMERS_CACHE":    cache_root / "huggingface" / "hub",
        "HF_DATASETS_CACHE":     cache_root / "huggingface" / "datasets",
        "TORCH_HOME":            cache_root / "torch",
        "WHISPER_CACHE":         cache_root / "whisper",
        "XDG_CACHE_HOME":        cache_root / "xdg",
    }

    for env_var, path in subdirs.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(env_var, str(path))

    # Redirect jieba (Chinese tokeniser bundled in F5-TTS) away from C:/Temp
    jieba_cache = cache_root / "jieba"
    jieba_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JIEBA_CACHE", str(jieba_cache / "jieba.cache"))
    try:
        import tempfile as _tempfile
        # jieba uses tempfile.gettempdir() — override via TMPDIR / TEMP / TMP
        os.environ.setdefault("TMPDIR", str(jieba_cache))
        os.environ.setdefault("TEMP",   str(jieba_cache))
        os.environ.setdefault("TMP",    str(jieba_cache))
    except Exception:
        pass

    # Redirect jieba (bundled tokeniser) away from system Temp
    jieba_dir = cache_root / 'jieba'
    jieba_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('JIEBA_CACHE', str(jieba_dir / 'jieba.cache'))
    os.environ.setdefault('TMPDIR', str(jieba_dir))
    os.environ.setdefault('TEMP',   str(jieba_dir))
    os.environ.setdefault('TMP',    str(jieba_dir))

    return str(cache_root)

import numpy as np
import soundfile as sf

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("IndicF5Pipeline")
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Audio Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

class AudioPreprocessor:
    """
    Prepares a raw reference audio file for IndicF5 inference.

    IndicF5 (built on F5-TTS) expects:
        • Sample rate  : 24 000 Hz
        • Channels     : Mono
        • Duration     : 3 – 15 seconds  (model uses up to ~15 s of context)
        • Amplitude    : normalised to [-1, 1]
        • Format       : WAV / PCM float32
    """

    TARGET_SR = 24_000          # IndicF5 native sample rate
    MIN_DURATION_SEC = 3.0
    MAX_DURATION_SEC = 6.0      # MUST be short (~6s) so the model has room left to generate long text!
    SILENCE_THRESHOLD_DB = -40  # dB below which a frame is treated as silence
    TOP_DB = 40                 # librosa trim parameter

    def __init__(self, denoise: bool = False):
        self.denoise = denoise

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _load_audio(path: str) -> tuple[np.ndarray, int]:
        """Load audio file → (float32 array, sample_rate)."""
        try:
            audio, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception as exc:
            raise RuntimeError(f"Cannot read audio file '{path}': {exc}") from exc
        return audio, sr

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        """Convert multi-channel audio to mono by averaging channels."""
        if audio.ndim == 2 and audio.shape[1] > 1:
            log.info("  Converting %d-channel audio → mono", audio.shape[1])
            audio = audio.mean(axis=1)
        else:
            audio = audio[:, 0] if audio.ndim == 2 else audio
        return audio

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio using librosa (high-quality sinc interpolation)."""
        if orig_sr == target_sr:
            return audio
        try:
            import librosa
            log.info("  Resampling %d Hz → %d Hz", orig_sr, target_sr)
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Fallback: scipy
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(orig_sr, target_sr)
            log.info("  Resampling (scipy) %d Hz → %d Hz", orig_sr, target_sr)
            return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)

    @staticmethod
    def _trim_silence(audio: np.ndarray, sr: int, top_db: int = 40) -> np.ndarray:
        """Trim leading / trailing silence."""
        try:
            import librosa
            trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
            log.info(
                "  Trimmed silence: %.2fs → %.2fs",
                len(audio) / sr,
                len(trimmed) / sr,
            )
            return trimmed
        except ImportError:
            # Simple energy-based trim
            rms_frame = 512
            hop = 256
            frames = [
                audio[i : i + rms_frame]
                for i in range(0, len(audio) - rms_frame, hop)
            ]
            energies = np.array([np.sqrt(np.mean(f**2)) for f in frames])
            threshold = 10 ** (AudioPreprocessor.SILENCE_THRESHOLD_DB / 20)
            nonsilent = np.where(energies > threshold)[0]
            if len(nonsilent) == 0:
                return audio
            start = nonsilent[0] * hop
            end = min((nonsilent[-1] + 1) * hop + rms_frame, len(audio))
            return audio[start:end]

    @staticmethod
    def _normalize(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        """Peak-normalize audio."""
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio * (target_peak / peak)
        return audio.astype(np.float32)

    @staticmethod
    def _clip_duration(audio: np.ndarray, sr: int, max_sec: float) -> np.ndarray:
        """Clip to maximum duration, keeping the best (middle) portion for voice."""
        max_samples = int(max_sec * sr)
        if len(audio) <= max_samples:
            return audio
        # Keep middle segment — usually cleaner than start/end
        mid = len(audio) // 2
        half = max_samples // 2
        start = max(0, mid - half)
        end = start + max_samples
        log.info(
            "  Clipping audio to %.1f s (model max context)",
            max_sec,
        )
        return audio[start:end]

    @staticmethod
    def _optional_denoise(audio: np.ndarray, sr: int) -> np.ndarray:
        """Optional spectral-subtraction denoising via noisereduce."""
        try:
            import noisereduce as nr
            log.info("  Applying noise reduction (noisereduce) …")
            return nr.reduce_noise(y=audio, sr=sr, stationary=False).astype(np.float32)
        except ImportError:
            log.warning(
                "  'noisereduce' not installed — skipping denoising. "
                "Install with: pip install noisereduce"
            )
            return audio

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, input_path: str, output_path: str) -> str:
        """
        Full preprocessing pipeline.

        Returns
        -------
        str
            Path to the preprocessed WAV file (ready for IndicF5).
        """
        log.info("━━━ STEP 1: Audio Preprocessing ━━━")
        log.info("  Input : %s", input_path)

        audio, sr = self._load_audio(input_path)
        audio = self._to_mono(audio)

        if self.denoise:
            audio = self._optional_denoise(audio, sr)

        audio = self._resample(audio, sr, self.TARGET_SR)
        audio = self._trim_silence(audio, self.TARGET_SR)
        audio = self._clip_duration(audio, self.TARGET_SR, self.MAX_DURATION_SEC)
        audio = self._normalize(audio)

        duration = len(audio) / self.TARGET_SR
        if duration < self.MIN_DURATION_SEC:
            raise ValueError(
                f"Reference audio is only {duration:.1f} s after preprocessing. "
                f"Please provide at least {self.MIN_DURATION_SEC} s of clear speech."
            )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        sf.write(output_path, audio, self.TARGET_SR, subtype="PCM_16")
        log.info("  Saved preprocessed audio → %s  (%.2f s)", output_path, duration)
        return output_path


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Reference Audio Transcription
# ══════════════════════════════════════════════════════════════════════════════

class ReferenceTranscriber:
    """
    Auto-transcribes the reference audio to obtain ref_text required by IndicF5.

    Priority:
        1. User-supplied text (skip transcription entirely)
        2. OpenAI Whisper (multilingual, works well for Indian languages)
        3. Faster-Whisper (CPU-friendly alternative)
    """

    def __init__(self, model_size: str = "medium"):
        self.model_size = model_size  # tiny | base | small | medium | large-v2

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        """Return transcript of the reference audio clip."""
        log.info("━━━ STEP 2: Transcribing Reference Audio ━━━")
        log.info("  (Reference text not provided; initializing Whisper ...)")
        text = self._try_whisper(audio_path, language)
        if text:
            log.info("  Transcript: %s", text)
            return text
        raise RuntimeError(
            "Transcription failed. Ensure Whisper is installed or pass --ref_text."
        )

    def _try_whisper(self, audio_path: str, language: str | None) -> str | None:
        try:
            import whisper
            log.info("  Using OpenAI Whisper (%s) …", self.model_size)
            model = whisper.load_model(self.model_size)
            opts = {"task": "transcribe"}
            if language:
                opts["language"] = language
            result = model.transcribe(audio_path, **opts)
            return result["text"].strip()
        except ImportError:
            pass
        try:
            from faster_whisper import WhisperModel
            log.info("  Using faster-whisper (%s) …", self.model_size)
            model = WhisperModel(self.model_size, compute_type="int8")
            segs, _ = model.transcribe(audio_path, language=language)
            return " ".join(s.text for s in segs).strip()
        except ImportError:
            pass
        return None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — IndicF5 Inference
# ══════════════════════════════════════════════════════════════════════════════


def _patch_indicf5_model_py(cache_root=None):
    # Removes torch.compile() from cached IndicF5 model.py.
    # torch.compile() inside __init__ breaks Transformers meta-tensor init,
    # causing: RuntimeError: Tensor.item() cannot be called on meta tensors
    import glob
    search_roots = [r for r in [
        cache_root,
        os.environ.get('HF_HOME'),
        os.environ.get('HUGGINGFACE_HUB_CACHE'),
    ] if r]
    needle_short = 'self.vocoder = torch.compile(load_vocoder('
    needle_full  = (
        'self.vocoder = torch.compile('
        'load_vocoder(vocoder_name="vocos", is_local=False, device=device))'
    )
    replacement  = (
        'self.vocoder = '
        'load_vocoder(vocoder_name="vocos", is_local=False, device=device)'
    )
    found = []
    for root in search_roots:
        import glob as _glob
        found += _glob.glob(
            str(Path(root) / '**' / 'transformers_modules'
                / 'ai4bharat' / 'IndicF5' / '*' / 'model.py'),
            recursive=True,
        )
    for fp in found:
        try:
            txt = Path(fp).read_text(encoding='utf-8')
            if needle_short in txt:
                Path(fp).write_text(
                    txt.replace(needle_full, replacement, 1),
                    encoding='utf-8',
                )
                log.info('  Patched torch.compile out of %s', fp)
            else:
                log.debug('  Already patched: %s', fp)
        except Exception as exc:
            log.warning('  Could not patch %s: %s', fp, exc)


class IndicF5VoiceCloner:
    """
    Wraps the AI4Bharat IndicF5 model for zero-shot voice cloning inference.

    The model API:
        audio = model(
            text,                  # target text to synthesize
            ref_audio_path=...,    # preprocessed reference wav
            ref_text=...,          # transcript of reference audio
        )
    Output: numpy int16 array at 24 000 Hz.
    """

    MODEL_ID = "ai4bharat/IndicF5"

    def __init__(self):
        self.model = None

    def load(self, hf_token: str | None = None):
        log.info("━━━ STEP 3: Loading IndicF5 Model ━━━")
        log.info("  Fetching %s from Hugging Face ...", self.MODEL_ID)

        # Resolve token: argument > HF_TOKEN env var > cached huggingface-cli login
        token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if token:
            log.info("  Using HuggingFace token for authentication.")
            # Inject token into environment so ALL hf_hub_download calls
            # inside model.py (vocab.txt, etc.) pick it up automatically
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGINGFACE_TOKEN"] = token
            try:
                from huggingface_hub import login as _hf_login
                _hf_login(token=token, add_to_git_credential=False)
                log.info("  Token registered with huggingface_hub session.")
            except Exception as _e:
                log.debug("  huggingface_hub.login() skipped: %s", _e)
        else:
            log.info("  No token provided - relying on cached login credentials.")

        # Patch model.py to remove torch.compile (meta-tensor / Windows fix)
        _patch_indicf5_model_py(cache_root=os.environ.get('HF_HOME'))

        try:
            import torch
            from transformers import AutoModel

            # ── Meta-tensor guard ───────────────────────────────────────────
            # Transformers initialises large models under a meta-device context
            # so __init__ runs with fake zero-memory tensors. Vocos/torchaudio
            # calls .any()/.item() on these tensors during MelSpectrogram init,
            # which is forbidden. We monkey-patch the offending check away.
            import torchaudio.functional as _taf
            _orig_melscale = _taf.melscale_fbanks
            def _safe_melscale(*a, **kw):
                fb = _orig_melscale(*a, **kw)
                # skip the meta-unsafe .any() check; return as-is
                return fb
            _taf.melscale_fbanks = _safe_melscale

            import torchaudio.transforms as _tat
            _orig_MelScale_init = _tat.MelScale.__init__
            def _safe_MelScale_init(self_ms, *a, **kw):
                orig = _taf.melscale_fbanks
                _taf.melscale_fbanks = _safe_melscale
                try:
                    _orig_MelScale_init(self_ms, *a, **kw)
                finally:
                    _taf.melscale_fbanks = orig
            _tat.MelScale.__init__ = _safe_MelScale_init
            # ────────────────────────────────────────────────────────────────

            self.model = AutoModel.from_pretrained(
                self.MODEL_ID,
                trust_remote_code=True,
                token=token,
            )

            # Restore originals after loading
            _taf.melscale_fbanks  = _orig_melscale
            _tat.MelScale.__init__ = _orig_MelScale_init

            # Re-patch now model.py is on disk (first-run download case)
            _patch_indicf5_model_py(cache_root=os.environ.get('HF_HOME'))
        except Exception as exc:
            err = str(exc)
            if "401" in err or "gated" in err.lower() or "unauthorized" in err.lower():
                raise RuntimeError(
                    "\n"
                    "╔══════════════════════════════════════════════════════════════╗\n"
                    "║  HUGGINGFACE AUTHENTICATION REQUIRED                        ║\n"
                    "║                                                              ║\n"
                    "║  IndicF5 is a gated model. Do these TWO things:             ║\n"
                    "║                                                              ║\n"
                    "║  1. Accept terms in your browser at:                        ║\n"
                    "║     https://huggingface.co/ai4bharat/IndicF5               ║\n"
                    "║     (click 'Agree and access repository')                   ║\n"
                    "║                                                              ║\n"
                    "║  2. Get a token at:                                         ║\n"
                    "║     https://huggingface.co/settings/tokens                 ║\n"
                    "║     Then authenticate via ONE of these options:             ║\n"
                    "║                                                              ║\n"
                    "║  Option A - pass token as CLI argument:                     ║\n"
                    "║     python indic_f5_voice_clone.py ...                     ║\n"
                    "║         --hf_token hf_xxxxxxxxxxxxxxxx                     ║\n"
                    "║                                                              ║\n"
                    "║  Option B - set environment variable (PowerShell):          ║\n"
                    "║     $env:HF_TOKEN = 'hf_xxxxxxxxxxxxxxxx'                  ║\n"
                    "║     python indic_f5_voice_clone.py ...                     ║\n"
                    "║                                                              ║\n"
                    "║  Option C - login once (token saved to disk):               ║\n"
                    "║     python -c \"from huggingface_hub import login; login()\"║\n"
                    "╚══════════════════════════════════════════════════════════════╝"
                ) from exc
            raise RuntimeError(
                f"Failed to load IndicF5: {exc}\n"
                "Make sure IndicF5 is installed:\n"
                "    pip install git+https://github.com/ai4bharat/IndicF5.git"
            ) from exc

    def synthesize(
        self,
        target_text: str,
        ref_audio_path: str,
        ref_text: str,
        hf_token: str | None = None,
    ) -> np.ndarray:
        """
        Run inference.

        Returns
        -------
        np.ndarray
            Float32 audio array at 24 000 Hz.
        """
        if self.model is None:
            self.load(hf_token=hf_token)

        log.info("━━━ STEP 4: Synthesizing Speech ━━━")
        log.info("  Target text : %s", target_text[:80] + ("…" if len(target_text) > 80 else ""))
        log.info("  Ref text    : %s", ref_text[:80] + ("…" if len(ref_text) > 80 else ""))

        # Chunk the text to handle long generations and avoid truncation
        import re
        # Smart split: splits by terminators but keeps the terminator attached to the preceding text
        sentences = re.split(r'(?<=[।॥.!?\n])\s+', target_text.strip())
        
        chunks = []
        current_chunk = ""
        # 120 chars is ideal when paired with a 5-6s reference audio
        MAX_CHUNK_LEN = 120 
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if not current_chunk:
                current_chunk = sentence
            elif len(current_chunk) + len(sentence) + 1 <= MAX_CHUNK_LEN:
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk)
                current_chunk = sentence
                
        if current_chunk:
            chunks.append(current_chunk)

        audio_pieces = []
        log.info("  Split text into %d chunks to avoid model truncation.", len(chunks))
        
        for i, chunk in enumerate(chunks, 1):
            if not chunk:
                continue
            log.info("  Processing chunk %d/%d: %s", i, len(chunks), chunk[:40] + ("..." if len(chunk) > 40 else ""))
            try:
                raw = self.model(
                    chunk,
                    ref_audio_path=ref_audio_path,
                    ref_text=ref_text,
                )
                
                # IndicF5 returns int16 → convert to float32
                chunk_audio = np.array(raw, dtype=np.float32)
                if chunk_audio.max() > 1.0 or chunk_audio.min() < -1.0:
                    chunk_audio = chunk_audio / 32768.0
                    
                audio_pieces.append(chunk_audio)
                
                # Optionally add a tiny 0.2s pause between sentences to prevent clipping
                pause = np.zeros(int(24000 * 0.2), dtype=np.float32)
                audio_pieces.append(pause)
                
            except Exception as e:
                log.error("  Error generating chunk %d: %s", i, e)

        if not audio_pieces:
            raise RuntimeError("Failed to generate any audio. Text might be invalid.")

        audio = np.concatenate(audio_pieces)
        log.info("  Synthesis complete — %.2f s of audio total", len(audio) / 24_000)
        return audio


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Post-Processing & Save
# ══════════════════════════════════════════════════════════════════════════════

class OutputProcessor:
    """Normalise, optionally resample, and write the synthesised audio."""

    SAMPLE_RATE = 24_000  # IndicF5 output sample rate

    def save(
        self,
        audio: np.ndarray,
        output_path: str,
        target_sr: int | None = None,
    ) -> str:
        log.info("━━━ STEP 5: Saving Output Audio ━━━")

        # Normalise
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio * (0.95 / peak)

        sr = self.SAMPLE_RATE
        if target_sr and target_sr != self.SAMPLE_RATE:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
                log.info("  Resampled output to %d Hz", sr)
            except ImportError:
                log.warning("  librosa not found — output stays at %d Hz", sr)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        sf.write(output_path, audio.astype(np.float32), sr)
        log.info("  Output saved → %s  (%d Hz, %.2f s)", output_path, sr, len(audio) / sr)
        return output_path


# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class IndicF5Pipeline:
    """
    End-to-end voice cloning pipeline.

    Usage
    -----
    pipeline = IndicF5Pipeline(denoise=True)
    pipeline.run(
        ref_audio   = "my_voice.wav",
        target_text = "नमस्ते! यह एक परीक्षण है।",
        output_path = "output/cloned.wav",
        ref_text    = None,   # None → auto-transcribe
        language    = "hi",   # ISO code for Whisper (optional)
    )
    """

    def __init__(
        self,
        denoise: bool = False,
        whisper_model: str = "medium",
        cache_dir: str | None = None,
    ):
        # Redirect all caches BEFORE importing torch / transformers / whisper
        self.cache_root = _setup_local_cache(cache_dir)
        log.info("Cache root: %s", self.cache_root)

        self.preprocessor = AudioPreprocessor(denoise=denoise)
        self.transcriber  = ReferenceTranscriber(model_size=whisper_model)
        self.cloner       = IndicF5VoiceCloner()
        self.output_proc  = OutputProcessor()

    def run(
        self,
        ref_audio: str,
        target_text: str,
        output_path: str = "output/cloned_voice.wav",
        ref_text: str | None = None,
        language: str | None = None,
        output_sr: int | None = None,
        hf_token: str | None = None,
    ) -> str:
        """
        Execute the full pipeline.

        Parameters
        ----------
        ref_audio    : Path to reference speaker audio (any common format).
        target_text  : Text to synthesize in the cloned voice.
        output_path  : Where to write the output WAV.
        ref_text     : Transcript of ref_audio. If None, auto-transcribed.
        language     : BCP-47 / ISO-639 language code for Whisper (e.g. "hi", "ta").
        output_sr    : Optional output sample rate (default: 24 000 Hz).
        hf_token     : HuggingFace access token for gated model download.

        Returns
        -------
        str
            Path to the generated audio file.
        """
        log.info("╔══════════════════════════════════════════════════╗")
        log.info("║       Indic Voice Cloning Pipeline               ║")
        log.info("╚══════════════════════════════════════════════════╝")

        # 1 — Preprocess reference audio
        preprocessed_path = str(
            Path(output_path).parent / "_ref_preprocessed.wav"
        )
        self.preprocessor.process(ref_audio, preprocessed_path)

        # 2 — Get reference transcript
        if ref_text and ref_text.strip():
            log.info("━━━ STEP 2: Bypassing Transcription ━━━")
            log.info("  Using provided ref_text: %s", ref_text[:100] + ("..." if len(ref_text) > 100 else ""))
        else:
            ref_text = self.transcriber.transcribe(preprocessed_path, language=language)

        # 3 & 4 — Load model and synthesise
        audio = self.cloner.synthesize(
            target_text=target_text,
            ref_audio_path=preprocessed_path,
            ref_text=ref_text,
            hf_token=hf_token,
        )

        # 5 — Post-process and save
        result = self.output_proc.save(audio, output_path, target_sr=output_sr)

        log.info("╔══════════════════════════════════════════════════╗")
        log.info("║  ✓  Done!  Output → %-28s ║", result[-28:])
        log.info("╚══════════════════════════════════════════════════╝")
        return result


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="IndicF5 End-to-End Voice Cloning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 WINDOWS USERS — READ THIS FIRST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PowerShell / CMD cannot reliably pass Indic Unicode text
as command-line arguments.  Use --text_file instead:

  Step 1 — Create text.txt in Notepad or VS Code,
           save it as UTF-8 (NOT UTF-8 with BOM).
           Write your Hindi/Tamil/etc. text inside.

  Step 2 — Run:
    python indic_f5_voice_clone.py \
        --ref_audio  my_speaker.wav \
        --text_file  text.txt \
        --language   hi

  Alternatively, enable UTF-8 in PowerShell first:
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $env:PYTHONUTF8 = "1"
    chcp 65001
  Then use --text normally.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Examples
--------
# Hindi via text file (recommended on Windows):
  python indic_f5_voice_clone.py \
      --ref_audio  speaker.wav \
      --text_file  hindi_text.txt \
      --language   hi

# Tamil with inline text (Linux/Mac or UTF-8 terminal):
  python indic_f5_voice_clone.py \
      --ref_audio  speaker.wav \
      --text       "வணக்கம்! இது ஒரு சோதனை." \
      --ref_text   "இந்த ஒலி கோப்பு என் குரலை குளோன் செய்கிறது." \
      --output     output/tamil_clone.wav

# With noise reduction:
  python indic_f5_voice_clone.py \
      --ref_audio  noisy_mic.wav \
      --text_file  text.txt \
      --language   pa \
      --denoise

Supported language codes: as bn gu hi kn ml mr or pa ta te
        """,
    )

    p.add_argument(
        "--ref_audio", required=True,
        help="Path to reference speaker audio file (wav/mp3/flac/ogg …)",
    )

    # Text input — mutually exclusive: inline string OR file path
    text_group = p.add_mutually_exclusive_group(required=True)
    text_group.add_argument(
        "--text",
        help=(
            "Target text to synthesize. "
            "NOTE: On Windows PowerShell/CMD, Indic scripts passed here are "
            "often corrupted by the terminal's encoding. Use --text_file instead."
        ),
    )
    text_group.add_argument(
        "--text_file",
        metavar="PATH",
        help=(
            "Path to a UTF-8 encoded .txt file containing the target text. "
            "RECOMMENDED on Windows for Devanagari / Tamil / Telugu etc."
        ),
    )

    p.add_argument(
        "--output", default="output/cloned_voice.wav",
        help="Output WAV path (default: output/cloned_voice.wav)",
    )
    p.add_argument(
        "--ref_text", default=None,
        help=(
            "Transcript of the reference audio clip. "
            "If omitted, Whisper will auto-transcribe it. "
            "Use --ref_text_file on Windows for Indic scripts."
        ),
    )
    p.add_argument(
        "--ref_text_file",
        metavar="PATH",
        default=None,
        help="Path to a UTF-8 .txt file with the reference audio transcript.",
    )
    p.add_argument(
        "--language", default=None,
        help="ISO language code for Whisper (hi, ta, te, bn, kn, ml, mr, gu, pa, or, as)",
    )
    p.add_argument(
        "--whisper_model", default="medium",
        choices=["tiny", "base", "small", "medium", "large-v2"],
        help="Whisper model size for auto-transcription (default: medium)",
    )
    p.add_argument(
        "--denoise", action="store_true",
        help="Apply noise reduction to reference audio before processing",
    )
    p.add_argument(
        "--output_sr", type=int, default=None,
        help="Resample output to this sample rate Hz (default: 24000)",
    )
    p.add_argument(
        "--hf_token", default=None,
        metavar="hf_xxx",
        help=(
            "HuggingFace access token for downloading the gated IndicF5 model. "
            "Get yours at https://huggingface.co/settings/tokens . "
            "Also accepts env var HF_TOKEN. "
            "You must first accept terms at https://huggingface.co/ai4bharat/IndicF5"
        ),
    )

    p.add_argument(
        "--cache_dir",
        default=None,
        metavar="PATH",
        help=(
            "Root folder for ALL downloaded model caches "
            "(HuggingFace, Whisper, PyTorch, etc.). "
            "Defaults to a 'cache/' subfolder next to this script. "
            "Use this to keep everything off the C: drive."
        ),
    )

    return p


def _read_utf8_file(path: str, label: str) -> str:
    """Read a UTF-8 text file, with clear error messaging."""
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        # Try with BOM variant
        try:
            text = Path(path).read_text(encoding="utf-8-sig").strip()
        except Exception as exc:
            raise ValueError(
                f"{label} file '{path}' is not valid UTF-8.\n"
                "Save the file from Notepad as 'UTF-8' (not 'ANSI') or use VS Code."
            ) from exc
    if not text:
        raise ValueError(f"{label} file '{path}' is empty.")
    return text


def _detect_garbled(text: str) -> bool:
    """Return True if the text looks like a Windows encoding casualty (mostly '?')."""
    if not text:
        return False
    question_marks = text.count("?")
    return question_marks / len(text) > 0.4   # >40% '?' chars → almost certainly garbled


def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── Resolve target text ──────────────────────────────────────────────────
    if args.text_file:
        target_text = _read_utf8_file(args.text_file, "Target text")
        log.info("Loaded target text from file: %s", args.text_file)
    else:
        target_text = args.text
        if _detect_garbled(target_text):
            log.error(
                "\n"
                "╔══════════════════════════════════════════════════════════════╗\n"
                "║  ERROR: Target text appears garbled (too many '?' chars).   ║\n"
                "║                                                              ║\n"
                "║  Your Windows terminal corrupted the Indic Unicode text.    ║\n"
                "║                                                              ║\n"
                "║  FIX — use --text_file instead:                             ║\n"
                "║    1. Save your text in a file, e.g. text.txt (UTF-8)      ║\n"
                "║    2. Run:                                                   ║\n"
                "║       python indic_f5_voice_clone.py \\                      ║\n"
                "║           --ref_audio my_speaker.wav \\                      ║\n"
                "║           --text_file text.txt \\                            ║\n"
                "║           --language  hi                                    ║\n"
                "║                                                              ║\n"
                "║  OR enable UTF-8 in PowerShell first:                       ║\n"
                "║       chcp 65001                                            ║\n"
                "║       $env:PYTHONUTF8 = '1'                                 ║\n"
                "╚══════════════════════════════════════════════════════════════╝"
            )
            sys.exit(1)

    # ── Resolve ref_text ─────────────────────────────────────────────────────
    ref_text = None
    if args.ref_text_file:
        ref_text = _read_utf8_file(args.ref_text_file, "Reference text")
        log.info("Loaded ref_text from file: %s", args.ref_text_file)
    elif args.ref_text:
        ref_text = args.ref_text
        if _detect_garbled(ref_text):
            log.warning(
                "ref_text appears garbled. Consider using --ref_text_file with a UTF-8 file."
            )
    else:
        # AUTO-DETECTION: Check if a .txt file exists next to the .wav file
        audio_path = Path(args.ref_audio)
        potential_txt = audio_path.with_suffix(".txt")
        if potential_txt.exists():
            try:
                ref_text = _read_utf8_file(str(potential_txt), "Auto-detected reference text")
                log.info("Auto-detected companion transcript: %s", potential_txt.name)
            except Exception as e:
                log.debug("Auto-detection failed for %s: %s", potential_txt, e)

    # ── Run pipeline ─────────────────────────────────────────────────────────
    pipeline = IndicF5Pipeline(
        denoise=args.denoise,
        whisper_model=args.whisper_model,
        cache_dir=args.cache_dir,
    )
    pipeline.run(
        ref_audio=args.ref_audio,
        target_text=target_text,
        output_path=args.output,
        ref_text=ref_text,
        language=args.language,
        output_sr=args.output_sr,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()