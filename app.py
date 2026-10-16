#!/usr/bin/env python3
# ------------------------------------------------------------
# OmniVoice Cross‑Platform Demo (Gradio UI)
# ------------------------------------------------------------
# Detect the best device (CUDA → MPS → CPU) and launch a Gradio
# web UI that works offline. The UI is identical on macOS and
# Windows, exposing the three generation modes:
#   • Voice cloning (ref audio + optional transcript)
#   • Voice design (speaker attribute prompt)
#   • Auto voice (random voice)
# ------------------------------------------------------------

import os
import sys
import platform
import shutil
import warnings
from datetime import datetime
from typing import Optional

# Suppress repetitive HuggingFace/transformers tokenizer/whisper warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

if getattr(sys, 'frozen', False):
    # Running as compiled bundle
    base_path = sys._MEIPASS
    os.environ['HF_HOME'] = os.path.join(base_path, 'huggingface')
    os.environ['HF_HUB_OFFLINE'] = '1'

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from omnivoice import OmniVoice

# ------------------------------------------------------------------
# 1️⃣ Device selection
# ------------------------------------------------------------------
def get_device() -> str:
    """Return the optimal device for OmniVoice.
    Priority: CUDA (Windows) → MPS (macOS Apple Silicon) → CPU.
    """
    if torch.cuda.is_available():
        return "cuda:0"
    if platform.system() == "Darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()
print(f"[OmniVoice] Using device: {DEVICE}")

# ------------------------------------------------------------------
# 2️⃣ Load the model (once, reused across requests)
# ------------------------------------------------------------------
MODEL = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map=DEVICE,
    dtype=torch.float16 if "cuda" in DEVICE else torch.float32,
)

# ------------------------------------------------------------------
# 2.5️⃣ Setup Data Directories & Helper Functions
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_VOICES_DIR = os.path.join(BASE_DIR, "saved_voices")
DATA_DIR = os.path.join(BASE_DIR, "data")
HISTORY_DIR = os.path.join(DATA_DIR, "history")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(SAVED_VOICES_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

import re
import hashlib

def clean_text(text: Optional[str]) -> Optional[str]:
    """Sanitize and clean text input by stripping whitespace and control characters."""
    if not text:
        return text
    # Remove unprintable/control characters
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # Collapse multiple spaces and line breaks cleanly
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r'\n\s*\n+', '\n', cleaned).strip()
    return cleaned

import subprocess

def convert_to_wav(audio_path: Optional[str], target_sr: int = 24000) -> Optional[str]:
    """Auto-converts .m4a, .mp3, .ogg, .flac, .aac to 24kHz mono PCM WAV."""
    if not audio_path or not os.path.exists(audio_path):
        return audio_path
    ext = os.path.splitext(audio_path)[1].lower()
    if ext == ".wav":
        return audio_path
    
    file_hash = hashlib.md5(f"{audio_path}_{os.path.getmtime(audio_path)}".encode()).hexdigest()[:10]
    out_wav = os.path.join(CACHE_DIR, f"conv_{file_hash}.wav")
    if not os.path.exists(out_wav):
        ffmpeg_bin = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
        cmd = [
            ffmpeg_bin, "-y", "-i", audio_path,
            "-ar", str(target_sr), "-ac", "1", out_wav
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return out_wav
        except Exception as err:
            print(f"[AudioConvert Warning] ffmpeg failed: {err}")
            return audio_path
    return out_wav

def get_optimized_ref_audio(audio_path: Optional[str], max_duration: float = 8.0) -> Optional[str]:
    """Automatically convert format, trim long reference audio and downmix to mono."""
    if not audio_path or not os.path.exists(audio_path):
        return audio_path
    # Ensure WAV format first
    audio_path = convert_to_wav(audio_path)
    try:
        info = sf.info(audio_path)
        duration = info.duration
        if duration > 12.0 or info.channels > 1:
            file_hash = hashlib.md5(f"{audio_path}_{os.path.getmtime(audio_path)}".encode()).hexdigest()[:10]
            cached_filename = f"trimmed_{file_hash}.wav"
            cached_path = os.path.join(CACHE_DIR, cached_filename)
            
            if not os.path.exists(cached_path):
                data, samplerate = sf.read(audio_path)
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                max_samples = int(min(max_duration, duration) * samplerate)
                trimmed_data = data[:max_samples]
                sf.write(cached_path, trimmed_data, samplerate)
            return cached_path
    except Exception as e:
        print(f"[SmartTrim Warning] Could not optimize audio: {e}")
    return audio_path

def get_history_files():
    if not os.path.exists(HISTORY_DIR):
        return []
    files = [f for f in os.listdir(HISTORY_DIR) if f.endswith(".wav")]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(HISTORY_DIR, x)), reverse=True)
    return files

def get_saved_voices():
    if not os.path.exists(SAVED_VOICES_DIR):
        return []
    files = [f for f in os.listdir(SAVED_VOICES_DIR) if f.endswith(".wav")]
    files.sort()
    return files

def save_new_voice(filepath, name, transcript=""):
    if not filepath or not name:
        return "Please provide both audio and a name."
    safe_name = "".join([c for c in name if c.isalnum() or c in [' ', '_', '-']]).strip()
    if not safe_name:
        return "Invalid name."
    
    # Auto convert .m4a or any uploaded format to WAV
    wav_path = convert_to_wav(filepath)
    out_path = os.path.join(SAVED_VOICES_DIR, safe_name + ".wav")
    shutil.copy2(wav_path, out_path)
    
    # Save transcript sidecar file
    text_path = os.path.join(SAVED_VOICES_DIR, safe_name + ".txt")
    cleaned_transcript = clean_text(transcript)
    if cleaned_transcript:
        try:
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(cleaned_transcript)
        except Exception as e:
            return f"Voice saved, but failed to save transcript: {e}"
    else:
        if os.path.exists(text_path):
            try:
                os.remove(text_path)
            except Exception:
                pass
    return f"Voice '{safe_name}' saved successfully!"

def load_saved_voice_and_text(sv):
    if not sv:
        return None, ""
    audio_path = os.path.abspath(os.path.join(SAVED_VOICES_DIR, sv))
    base_name = os.path.splitext(sv)[0]
    text_path = os.path.join(SAVED_VOICES_DIR, base_name + ".txt")
    transcript = ""
    if os.path.exists(text_path):
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                transcript = f.read().strip()
        except Exception as e:
            print(f"Error reading transcript: {e}")
    return audio_path, transcript

def update_saved_transcript(filename, new_transcript):
    if not filename:
        return "Please select a voice first."
    base_name = os.path.splitext(filename)[0]
    text_path = os.path.join(SAVED_VOICES_DIR, base_name + ".txt")
    cleaned = clean_text(new_transcript)
    if cleaned:
        try:
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(cleaned)
            return f"Transcript for '{base_name}' updated successfully!"
        except Exception as e:
            return f"Error updating transcript: {e}"
    else:
        if os.path.exists(text_path):
            try:
                os.remove(text_path)
            except Exception:
                pass
        return f"Transcript for '{base_name}' cleared."

def delete_history_item(filename):
    if not filename:
        return gr.Dropdown(choices=get_history_files())
    path = os.path.join(HISTORY_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
    return gr.Dropdown(choices=get_history_files(), value=None)

def delete_all_history():
    if os.path.exists(HISTORY_DIR):
        for f in os.listdir(HISTORY_DIR):
            if f.endswith(".wav"):
                os.remove(os.path.join(HISTORY_DIR, f))
    return gr.Dropdown(choices=get_history_files(), value=None)

def delete_saved_voice(filename):
    if not filename:
        return gr.Dropdown(choices=get_saved_voices())
    path = os.path.join(SAVED_VOICES_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
        # Delete sidecar transcript
        base_name = os.path.splitext(filename)[0]
        text_path = os.path.join(SAVED_VOICES_DIR, base_name + ".txt")
        if os.path.exists(text_path):
            try:
                os.remove(text_path)
            except Exception:
                pass
    return gr.Dropdown(choices=get_saved_voices(), value=None)

# ------------------------------------------------------------------
# 3️⃣ Generation helper – shared by all UI functions
# ------------------------------------------------------------------
@torch.inference_mode()
def generate_tts(
    text: str,
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    instruct: Optional[str] = None,
    speed: float = 0.95,
    num_step: int = 32,
) -> str:
    """Generate speech with torch.inference_mode(), full audio-text alignment, and text cleaning."""
    cleaned_text = clean_text(text)
    if not cleaned_text:
        raise gr.Error("Please enter text to synthesize.")
        
    cleaned_ref_text = clean_text(ref_text)

    kwargs = {"text": cleaned_text, "speed": speed, "num_step": num_step}
    if ref_audio:
        ref_audio = convert_to_wav(ref_audio)
        kwargs["ref_audio"] = ref_audio
        if cleaned_ref_text:
            kwargs["ref_text"] = cleaned_ref_text
    if instruct:
        kwargs["instruct"] = clean_text(instruct)
    
    audio = MODEL.generate(**kwargs)
    
    # Memory cache release for device stability
    if DEVICE == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    elif "cuda" in str(DEVICE) and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    now = datetime.now()
    month_lower = now.strftime("%b").lower()
    date_str = now.strftime(f"%d_{month_lower}_%Y_%I:%M%p")
    filename = f"Junovoice_{date_str}.wav"
    out_path = os.path.join(HISTORY_DIR, filename)
    
    counter = 1
    while os.path.exists(out_path):
        filename = f"Junovoice_{date_str}_{counter}.wav"
        out_path = os.path.join(HISTORY_DIR, filename)
        counter += 1

    sf.write(out_path, audio[0], 24000)
    return out_path

# ------------------------------------------------------------------
# 4️⃣ Gradio UI definition
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 4️⃣ Gradio UI definition
# ------------------------------------------------------------------
def build_interface() -> gr.Blocks:
    custom_css = """
body, html { 
    margin: 0 !important; 
    padding: 0 !important; 
    height: 100% !important; 
    background: radial-gradient(circle at 15% 50%, rgba(76, 29, 149, 0.4), transparent 50%),
                radial-gradient(circle at 85% 30%, rgba(14, 165, 233, 0.4), transparent 50%),
                radial-gradient(circle at 50% 80%, rgba(16, 185, 129, 0.2), transparent 50%),
                #121214 !important; 
    background-attachment: fixed !important; 
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif !important; 
    color: #f5f5f7 !important; 
    overflow: hidden !important; 
    -webkit-font-smoothing: antialiased !important;
}

.gradio-container { 
    max-width: 1400px !important; 
    width: 100% !important; 
    margin: 0 auto !important;
    background: transparent !important; 
    border: none !important; 
    height: 100vh !important; 
    display: flex !important; 
    flex-direction: column !important; 
    justify-content: center !important; 
    align-items: center !important;
    padding: 0 !important;
}

#glass-app { 
    width: 95% !important;
    max-width: 1200px !important;
    background: rgba(30, 30, 35, 0.35) !important; 
    backdrop-filter: blur(40px) saturate(200%) !important; 
    -webkit-backdrop-filter: blur(40px) saturate(200%) !important; 
    border: 1px solid rgba(255, 255, 255, 0.1) !important; 
    border-radius: 20px !important; 
    box-shadow: 0 30px 60px rgba(0,0,0,0.4), 
                0 0 0 1px rgba(255,255,255,0.05) inset,
                0 15px 25px rgba(0,0,0,0.2) !important; 
    overflow-y: auto !important; 
    overflow-x: hidden !important;
    padding: 24px 35px !important; 
    margin: 0 auto !important; 
    max-height: 96vh !important; 
}

/* Scrollbar styling for Mac OS feel */
#glass-app::-webkit-scrollbar { width: 6px; } /* Make scrollbar slim but visible */
#glass-app::-webkit-scrollbar-track { background: transparent; }
#glass-app::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.2); border-radius: 4px; }
#glass-app::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.3); }

/* Typography */
h1, h2, h3, h4, h5, h6 { color: #ffffff !important; letter-spacing: -0.02em !important; }

/* Panels and forms */
.tabs, .form, .block { background: transparent !important; border: none !important; box-shadow: none !important; }

/* Segmented Control for Tabs (Mac OS style) */
.tab-nav { 
    background: rgba(0, 0, 0, 0.2) !important; 
    border: 1px solid rgba(255,255,255,0.05) !important; 
    border-radius: 12px !important; 
    padding: 4px !important; 
    margin-bottom: 16px !important; 
    display: inline-flex !important; 
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.3) !important;
}
.tab-nav > button { 
    border: none !important; 
    background: transparent !important; 
    color: #a1a1a6 !important; 
    font-weight: 500 !important; 
    font-size: 14px !important;
    border-radius: 8px !important; 
    padding: 6px 24px !important; 
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important; 
}
.tab-nav > button.selected { 
    background: rgba(255,255,255,0.15) !important; 
    color: #ffffff !important; 
    box-shadow: 0 2px 8px rgba(0,0,0,0.2), inset 0 1px 1px rgba(255,255,255,0.1) !important; 
}
.tab-nav > button:hover:not(.selected) { background: rgba(255,255,255,0.08) !important; color: #e5e5e7 !important; }

/* Inputs and Textareas */
textarea, input[type="text"] { 
    background: rgba(0, 0, 0, 0.25) !important; 
    border: 1px solid rgba(255, 255, 255, 0.08) !important; 
    border-radius: 12px !important; 
    color: #ffffff !important; 
    transition: all 0.3s ease !important; 
    box-shadow: inset 0 2px 5px rgba(0,0,0,0.2) !important; 
    padding: 12px 16px !important; 
    font-size: 15px !important;
    line-height: 1.5 !important;
}
textarea:focus, input[type="text"]:focus { 
    border-color: rgba(10, 132, 255, 0.6) !important; 
    box-shadow: 0 0 0 4px rgba(10, 132, 255, 0.2), inset 0 2px 5px rgba(0,0,0,0.2) !important; 
    background: rgba(0, 0, 0, 0.35) !important; 
    outline: none !important; 
}

/* Audio/Upload Component */
div[data-testid="audio"], .audio-container, .upload-container { 
    background: rgba(0, 0, 0, 0.2) !important; 
    border: 2px dashed rgba(255, 255, 255, 0.1) !important; 
    border-radius: 16px !important; 
    overflow: hidden !important; 
    transition: all 0.3s ease !important; 
    box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.2) !important;
}
div[data-testid="audio"]:hover, .audio-container:hover, .upload-container:hover { 
    border-color: rgba(10, 132, 255, 0.5) !important; 
    background: rgba(10, 132, 255, 0.05) !important; 
}
div[data-testid="audio"] .upload {
    padding: 40px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
}
div[data-testid="audio"] .upload-text, div[data-testid="audio"] span, div[data-testid="audio"] svg {
    color: #a1a1a6 !important;
    font-weight: 500 !important;
}
div[data-testid="audio"]:hover .upload-text, div[data-testid="audio"]:hover span, div[data-testid="audio"]:hover svg {
    color: #309cff !important;
}
div[data-testid="audio"] button { 
    border-radius: 10px !important; 
    background: rgba(255,255,255,0.08) !important; 
    border: 1px solid rgba(255,255,255,0.1) !important; 
    color: #fff !important; 
    transition: all 0.2s ease !important; 
}
div[data-testid="audio"] button:hover { 
    background: rgba(255,255,255,0.15) !important; 
    transform: scale(1.02); 
}
.file-preview { border: none !important; background: transparent !important; }

/* Labels */
.form > div > label > span { 
    color: #a1a1a6 !important; 
    font-size: 12px !important; 
    font-weight: 600 !important; 
    text-transform: uppercase !important; 
    letter-spacing: 0.8px !important; 
    margin-bottom: 8px !important; 
    display: block !important; 
}

/* Primary Button (Mac OS Accent) */
button.primary { 
    background: linear-gradient(180deg, #148eff 0%, #007aff 100%) !important; 
    border: 1px solid rgba(0,0,0,0.1) !important; 
    border-radius: 12px !important; 
    color: white !important; 
    font-weight: 600 !important; 
    font-size: 15px !important; 
    padding: 12px 24px !important; 
    box-shadow: 0 4px 12px rgba(0, 122, 255, 0.3), inset 0 1px 1px rgba(255,255,255,0.2) !important; 
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important; 
    margin-top: 15px !important; 
    width: 100% !important; 
    cursor: pointer !important;
}
button.primary:hover { 
    transform: translateY(-1px) !important; 
    box-shadow: 0 6px 16px rgba(0, 122, 255, 0.4), inset 0 1px 1px rgba(255,255,255,0.3) !important; 
    background: linear-gradient(180deg, #309cff 0%, #0a84ff 100%) !important; 
}
button.primary:active { 
    transform: translateY(1px) !important; 
    box-shadow: 0 2px 8px rgba(0, 122, 255, 0.3) !important; 
}

/* Sliders */
input[type="range"] { accent-color: #0a84ff !important; }

/* Social Buttons */
.social-btn { 
    display: inline-flex; 
    align-items: center; 
    gap: 8px; 
    padding: 8px 16px; 
    border-radius: 12px; 
    text-decoration: none; 
    font-weight: 500; 
    font-size: 13px; 
    transition: all 0.3s ease; 
    background: rgba(255,255,255,0.05); 
    border: 1px solid rgba(255,255,255,0.08); 
}
.social-btn:hover { 
    background: rgba(255,255,255,0.1); 
    transform: translateY(-1px); 
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

/* Hide footer and built-with Gradio */
footer { display: none !important; opacity: 0 !important; visibility: hidden !important; }
.gradio-container > .prose { display: none !important; }
.meta-text { display: none !important; }

/* Light Mode Overrides */
@media (prefers-color-scheme: light) {
    body, html {
        background: radial-gradient(circle at 15% 50%, rgba(76, 29, 149, 0.08), transparent 50%),
                    radial-gradient(circle at 85% 30%, rgba(14, 165, 233, 0.08), transparent 50%),
                    radial-gradient(circle at 50% 80%, rgba(16, 185, 129, 0.04), transparent 50%),
                    #f5f5f7 !important;
        color: #1d1d1f !important;
    }
    #glass-app {
        background: rgba(255, 255, 255, 0.7) !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
        box-shadow: 0 20px 40px rgba(0,0,0,0.08), 
                    0 0 0 1px rgba(255,255,255,0.5) inset,
                    0 10px 20px rgba(0,0,0,0.04) !important;
    }
    #glass-app::-webkit-scrollbar-thumb { background: rgba(0, 0, 0, 0.15); }
    #glass-app::-webkit-scrollbar-thumb:hover { background: rgba(0, 0, 0, 0.25); }
    
    h1, h2, h3, h4, h5, h6 { color: #1d1d1f !important; }
    
    .tab-nav {
        background: rgba(0, 0, 0, 0.04) !important;
        border: 1px solid rgba(0,0,0,0.05) !important;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.05) !important;
    }
    .tab-nav > button { color: #6e6e73 !important; }
    .tab-nav > button.selected {
        background: #ffffff !important;
        color: #1d1d1f !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04) !important;
    }
    .tab-nav > button:hover:not(.selected) { background: rgba(0,0,0,0.03) !important; color: #1d1d1f !important; }
    
    textarea, input[type="text"] {
        background: rgba(255, 255, 255, 0.8) !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
        color: #1d1d1f !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.02) !important;
    }
    textarea:focus, input[type="text"]:focus {
        border-color: #0a84ff !important;
        box-shadow: 0 0 0 4px rgba(10, 132, 255, 0.15), inset 0 2px 4px rgba(0,0,0,0.02) !important;
        background: #ffffff !important;
    }
    
    div[data-testid="audio"], .audio-container, .upload-container {
        background: rgba(255, 255, 255, 0.6) !important;
        border: 2px dashed rgba(0, 0, 0, 0.15) !important;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.02) !important;
    }
    div[data-testid="audio"]:hover, .audio-container:hover, .upload-container:hover {
        border-color: #0a84ff !important;
        background: rgba(10, 132, 255, 0.03) !important;
    }
    div[data-testid="audio"] .upload-text, div[data-testid="audio"] span, div[data-testid="audio"] svg {
        color: #6e6e73 !important;
    }
    div[data-testid="audio"]:hover .upload-text, div[data-testid="audio"]:hover span, div[data-testid="audio"]:hover svg {
        color: #0a84ff !important;
    }
    div[data-testid="audio"] button {
        background: rgba(0,0,0,0.04) !important;
        border: 1px solid rgba(0,0,0,0.1) !important;
        color: #1d1d1f !important;
    }
    div[data-testid="audio"] button:hover {
        background: rgba(0,0,0,0.08) !important;
    }
    
    .form > div > label > span { color: #6e6e73 !important; }
    
    .social-btn {
        background: rgba(0,0,0,0.03) !important;
        border: 1px solid rgba(0,0,0,0.08) !important;
        color: #1d1d1f !important;
    }
    .social-btn:hover { background: rgba(0,0,0,0.06) !important; }
    
    .brand-title { color: #1d1d1f !important; text-shadow: 0 2px 10px rgba(0,0,0,0.05) !important; }
    .brand-subtitle-accent { color: #1d1d1f !important; }
}
"""

    brand_html = """
<div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid rgba(255,255,255,0.08); animation: fadeIn 0.8s ease-out;">
    <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    
    <!-- Left: Logo & Title -->
    <div style="display: flex; align-items: center; gap: 12px;">
        <div style="display: flex; align-items: center; justify-content: center; width: 42px; height: 42px; background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.02)); border: 1px solid rgba(255,255,255,0.15); border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.2); backdrop-filter: blur(10px);">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="url(#blueGrad)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <defs>
                    <linearGradient id="blueGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#309cff" />
                        <stop offset="100%" stop-color="#007aff" />
                    </linearGradient>
                </defs>
                <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"></path>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                <line x1="12" y1="19" x2="12" y2="22"></line>
                <line x1="8" y1="22" x2="16" y2="22"></line>
            </svg>
        </div>
        <div style="text-align: left;">
            <h1 class="brand-title" style="font-size: 22px; font-weight: 700; letter-spacing: -0.02em; margin: 0 0 2px 0; color: #ffffff; text-shadow: 0 2px 10px rgba(0,0,0,0.3);">
                JunoVoice
            </h1>
            <p style="font-size: 12px; font-weight: 400; color: #a1a1a6; margin: 0; letter-spacing: 0.01em;">
                Engine by <span class="brand-subtitle-accent" style="color: #ffffff; font-weight: 500;">Sayed Johon</span>
            </p>
        </div>
    </div>

    <!-- Right: Links -->
    <div style="display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; align-items: center;">
        <a href="https://peeai.com" target="_blank" class="social-btn" style="color: #309cff; padding: 6px 12px; font-size: 12px; border-color: rgba(0, 122, 255, 0.3); background: rgba(0, 122, 255, 0.1);">
            <svg style="vertical-align: middle; width: 14px; height: 14px;" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M12 2L2 22h20L12 2zm0 3.8l7.2 14.2H4.8L12 5.8z"/></svg>
            Learn AI
        </a>
        <a href="https://www.youtube.com/@junoverseai" target="_blank" class="social-btn" style="color: #ff453a; padding: 6px 12px; font-size: 12px;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.5 12 3.5 12 3.5s-7.505 0-9.377.55a3.016 3.016 0 0 0-2.122 2.136C0 8.057 0 12 0 12s0 3.943.501 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.55 9.377.55 9.377.55s7.505 0 9.377-.55a3.016 3.016 0 0 0 2.122-2.136C24 15.943 24 12 24 12s0-3.943-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/></svg>
            YouTube
        </a>
        <a href="https://facebook.com/sjohon" target="_blank" class="social-btn" style="color: #0a84ff; padding: 6px 12px; font-size: 12px;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.469h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.469h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/></svg>
            Facebook
        </a>
        <a href="https://wa.me/+8801972072408" target="_blank" class="social-btn" style="color: #32d74b; padding: 6px 12px; font-size: 12px;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51a5.8 5.8 0 0 0-.57-.01c-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347m-5.421 7.403h-.004a9.87 9.87 0 0 1-5.031-1.378l-.361-.214-3.741.982.998-3.648-.235-.374a9.86 9.86 0 0 1-1.51-5.26c.001-5.45 4.436-9.884 9.888-9.884 2.64 0 5.122 1.03 6.988 2.898a9.825 9.825 0 0 1 2.893 6.994c-.003 5.45-4.437 9.885-9.885 9.884m8.413-18.297A11.815 11.815 0 0 0 12.05 0C5.495 0 .16 5.335.157 11.892c0 2.096.547 4.142 1.588 5.945L.057 24l6.305-1.654a11.882 11.882 0 0 0 5.683 1.448h.005c6.554 0 11.89-5.335 11.893-11.893a11.821 11.821 0 0 0-3.48-8.413Z"/></svg>
            WhatsApp
        </a>
        <a href="https://t.me/junoverseai" target="_blank" class="social-btn" style="color: #5e5ce6; padding: 6px 12px; font-size: 12px;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm5.894 8.221l-1.97 9.28c-.145.658-.537.818-1.084.508l-3-2.21-1.446 1.394c-.14.18-.357.295-.6.295-.002 0-.003 0-.005 0l.213-3.054 5.56-5.022c.24-.213-.054-.334-.373-.121l-6.869 4.326-2.96-.924c-.64-.203-.658-.64.135-.954l11.566-4.458c.538-.196 1.006.128.833.94z"/></svg>
            Telegram
        </a>
    </div>
</div>
"""

    with gr.Blocks(title="JunoVoice") as demo:
        with gr.Column(elem_id="glass-app"):
            gr.HTML(brand_html)

            with gr.Row():
                speed = gr.Slider(0.5, 2.0, value=0.95, label="Speed factor (>1 faster, <1 slower)")
                steps = gr.Slider(16, 64, step=8, value=32, label="Diffusion steps")

            with gr.Tabs():
                # Voice Cloning
                with gr.TabItem("Voice Cloning"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            txt = gr.Textbox(label="Text to synthesize", placeholder="Type what you want JunoVoice to say...", lines=2)
                            btn = gr.Button("Generate Voice", variant="primary")
                            out = gr.Audio(label="Generated speech")
                        with gr.Column(scale=1):
                            use_saved_voice = gr.Dropdown(label="Load Saved Voice", choices=get_saved_voices(), value="10S - Johon- And there is a massive giveaway at the end of the video.wav", interactive=True, allow_custom_value=True)
                            
                            gr.Markdown("---")
                            gr.Markdown("### Upload New Reference")
                            ref = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference audio (optional)")
                            
                            with gr.Row():
                                save_voice_name_clone = gr.Textbox(label="Name to save voice as", placeholder="e.g. My Custom Voice")
                                save_voice_btn_clone = gr.Button("Save to Voice Library")
                            save_voice_status_clone = gr.Markdown()
                            
                            gr.Markdown("---")
                            ref_t = gr.Textbox(label="Reference transcript (optional)")
                            
                # Voice Design
                with gr.TabItem("Voice Design"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            txt_design = gr.Textbox(label="Text to synthesize", placeholder="Type what you want JunoVoice to say...", lines=2)
                            btn_design = gr.Button("Generate Voice", variant="primary")
                            out_design = gr.Audio(label="Generated speech")
                        with gr.Column(scale=1):
                            instr = gr.Textbox(label="Speaker attributes (e.g. `female, low pitch, british accent`)", lines=4)
                            
                # Auto Voice
                with gr.TabItem("Auto Voice"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            txt_auto = gr.Textbox(label="Text to synthesize", placeholder="Type what you want JunoVoice to say...", lines=3)
                            btn_auto = gr.Button("Generate Voice", variant="primary")
                        with gr.Column(scale=1):
                            out_auto = gr.Audio(label="Generated speech")
                            
                # Voice Library
                with gr.TabItem("Voice Library"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Save a New Voice Sample")
                            new_voice_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Upload/Record Voice")
                            new_voice_name = gr.Textbox(label="Voice Name (e.g. John Doe)")
                            new_voice_text = gr.Textbox(label="Reference transcript (optional)", placeholder="e.g. The exact words spoken in the reference audio", lines=2)
                            save_voice_btn = gr.Button("Save Voice", variant="primary")
                            save_voice_status = gr.Markdown()
                            
                        with gr.Column(scale=1):
                            gr.Markdown("### Saved Voices")
                            saved_voice_dropdown = gr.Dropdown(label="Select Voice", choices=get_saved_voices(), allow_custom_value=True)
                            refresh_voices_btn = gr.Button("🔄 Refresh List")
                            saved_voice_playback = gr.Audio(label="Playback", interactive=False)
                            saved_voice_text = gr.Textbox(label="Saved transcript (editable)", interactive=True, lines=2)
                            with gr.Row():
                                update_voice_transcript_btn = gr.Button("💾 Update Transcript")
                                delete_voice_btn = gr.Button("🗑️ Delete Selected Voice")
                            voice_lib_status = gr.Markdown()
                            
                # History
                with gr.TabItem("History"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Generation History")
                            history_dropdown = gr.Dropdown(label="Select History Item", choices=get_history_files())
                            refresh_history_btn = gr.Button("🔄 Refresh History")
                        with gr.Column(scale=1):
                            history_audio = gr.Audio(label="Playback", interactive=False)
                            history_download = gr.File(label="Download File", interactive=False)
                            with gr.Row():
                                delete_item_btn = gr.Button("🗑️ Delete Selected")
                                delete_all_btn = gr.Button("🚨 Delete All History")

            # ---------------------------
            # Event Wiring
            # ---------------------------
            # Voice Cloning Generation
            def clone_fn(t, sv, r, rt, sp, st):
                actual_ref = r if r else (os.path.join(SAVED_VOICES_DIR, sv) if sv else None)
                out_path = generate_tts(t, ref_audio=actual_ref, ref_text=rt, speed=sp, num_step=st)
                return out_path, gr.Dropdown(choices=get_history_files(), value=os.path.basename(out_path))
                
            btn.click(
                fn=clone_fn,
                inputs=[txt, use_saved_voice, ref, ref_t, speed, steps],
                outputs=[out, history_dropdown]
            )

            # Load saved voice and transcript into UI elements
            use_saved_voice.change(
                fn=load_saved_voice_and_text,
                inputs=use_saved_voice,
                outputs=[ref, ref_t]
            )

            # In-line save from cloning tab
            save_voice_btn_clone.click(
                fn=save_new_voice,
                inputs=[ref, save_voice_name_clone, ref_t],
                outputs=save_voice_status_clone
            ).then(
                fn=lambda: gr.Dropdown(choices=get_saved_voices(), value=None),
                outputs=use_saved_voice
            ).then(
                fn=lambda: gr.Dropdown(choices=get_saved_voices(), value=None),
                outputs=saved_voice_dropdown
            )

            # Voice Design Generation
            def design_fn(t, i, sp, st):
                out_path = generate_tts(t, instruct=i, speed=sp, num_step=st)
                return out_path, gr.Dropdown(choices=get_history_files(), value=os.path.basename(out_path))

            btn_design.click(
                fn=design_fn,
                inputs=[txt_design, instr, speed, steps],
                outputs=[out_design, history_dropdown]
            )

            # Auto Voice Generation
            def auto_fn(t, sp, st):
                out_path = generate_tts(t, speed=sp, num_step=st)
                return out_path, gr.Dropdown(choices=get_history_files(), value=os.path.basename(out_path))

            btn_auto.click(
                fn=auto_fn,
                inputs=[txt_auto, speed, steps],
                outputs=[out_auto, history_dropdown]
            )

            # Voice Library Events
            save_voice_btn.click(
                fn=save_new_voice,
                inputs=[new_voice_audio, new_voice_name, new_voice_text],
                outputs=save_voice_status
            ).then(
                fn=lambda: gr.Dropdown(choices=get_saved_voices(), value=None),
                outputs=saved_voice_dropdown
            ).then(
                fn=lambda: gr.Dropdown(choices=get_saved_voices(), value=None),
                outputs=use_saved_voice
            )
            
            refresh_voices_btn.click(
                fn=lambda: gr.Dropdown(choices=get_saved_voices(), value=None),
                outputs=saved_voice_dropdown
            ).then(
                fn=lambda: gr.Dropdown(choices=get_saved_voices(), value=None),
                outputs=use_saved_voice
            )
            
            saved_voice_dropdown.change(
                fn=load_saved_voice_and_text,
                inputs=saved_voice_dropdown,
                outputs=[saved_voice_playback, saved_voice_text]
            )
            
            update_voice_transcript_btn.click(
                fn=update_saved_transcript,
                inputs=[saved_voice_dropdown, saved_voice_text],
                outputs=voice_lib_status
            )
            
            delete_voice_btn.click(
                fn=delete_saved_voice,
                inputs=saved_voice_dropdown,
                outputs=saved_voice_dropdown
            ).then(
                fn=lambda: gr.Dropdown(choices=get_saved_voices(), value=None),
                outputs=use_saved_voice
            ).then(
                fn=lambda: (None, ""),
                outputs=[saved_voice_playback, saved_voice_text]
            )

            # History Events
            refresh_history_btn.click(
                fn=lambda: gr.Dropdown(choices=get_history_files(), value=None),
                outputs=history_dropdown
            )
            
            history_dropdown.change(
                fn=lambda h: (os.path.join(HISTORY_DIR, h), os.path.join(HISTORY_DIR, h)) if h else (None, None),
                inputs=history_dropdown,
                outputs=[history_audio, history_download]
            )
            
            delete_item_btn.click(
                fn=delete_history_item,
                inputs=history_dropdown,
                outputs=history_dropdown
            ).then(
                fn=lambda: (None, None),
                outputs=[history_audio, history_download]
            )
            
            delete_all_btn.click(
                fn=delete_all_history,
                outputs=history_dropdown
            ).then(
                fn=lambda: (None, None),
                outputs=[history_audio, history_download]
            )

    return demo, custom_css

# ------------------------------------------------------------------
# 5️⃣ Launch
# ------------------------------------------------------------------
if __name__ == "__main__":
    demo, custom_css = build_interface()
    favicon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.svg")
    is_colab = "google.colab" in sys.modules or "COLAB_GPU" in os.environ
    port = int(os.environ.get("PORT", 7860 if is_colab else 9090))
    share = is_colab or "--share" in sys.argv
    print(f"[OmniVoice] Launching studio (Colab={is_colab}, Share={share}, Port={port})...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        css=custom_css,
        favicon_path=favicon_path if os.path.exists(favicon_path) else None,
        allowed_paths=[BASE_DIR, SAVED_VOICES_DIR]
    )
