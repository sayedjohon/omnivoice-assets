#!/usr/bin/env python3
"""
Pre-download and cache the OmniVoice model weights during setup.
Run during installation so the web UI starts instantly without waiting.
"""
import sys
import torch
from omnivoice import OmniVoice

def main():
    print("=" * 65)
    print("🎙️ Pre-caching OmniVoice Model Weights (k2-fsa/OmniVoice)...")
    print("=" * 65)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if "cuda" in device else torch.float32
    print(f"Target Device: {device} | Precision: {dtype}")
    
    OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=device,
        dtype=dtype,
    )
    print("\n✅ OmniVoice weights successfully pre-downloaded & cached!")

if __name__ == "__main__":
    main()
