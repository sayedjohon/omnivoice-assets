#!/usr/bin/env python3
"""
OmniVoice Community Colab Launcher
One-click runner for Google Colab GPU environment.
"""

import os
import sys
import subprocess

def main():
    print("=" * 60)
    print("🎙️ OmniVoice Neural Voice Studio — Colab Runner")
    print("=" * 60)
    
    # Check PyTorch & GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)} (VRAM: {round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)} GB)")
        else:
            print("⚠️ Warning: No CUDA GPU detected. Running on CPU may be slow.")
    except ImportError:
        pass

    # Launch app with public share link enabled
    os.environ["PORT"] = "7860"
    from app import build_interface, BASE_DIR, SAVED_VOICES_DIR
    demo, custom_css = build_interface()
    favicon_path = os.path.join(BASE_DIR, "icon.svg")
    
    print("\n🚀 Starting OmniVoice Studio with public share link...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        css=custom_css,
        favicon_path=favicon_path if os.path.exists(favicon_path) else None,
        allowed_paths=[BASE_DIR, SAVED_VOICES_DIR]
    )

if __name__ == "__main__":
    main()
