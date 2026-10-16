# 🎙️ OmniVoice Assets & Neural Studio

High-fidelity Multilingual & Bengali Zero-Shot Voice Cloning Studio powered by `k2-fsa/OmniVoice`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sayedjohon/omnivoice-assets/blob/main/OmniVoice_Community_Studio.ipynb)

---

## ⚡ Quick Start on Google Colab (One-Click)

Open any Google Colab notebook, set hardware accelerator to **T4 GPU**, and run:

```bash
# 1. Clone repository
!git clone https://github.com/sayedjohon/omnivoice-assets.git
%cd omnivoice-assets

# 2. Install requirements
!pip install -q -r requirements.txt

# 3. Launch Web Studio with public link
!python run_colab.py
```

Click the generated `https://xxxx.gradio.live` public link to open the custom Web Studio in your browser!

---

## 📁 Preloaded Voice Profiles (`saved_voices/`)

| Voice Name | Description | Reference Sample | Reference Transcript |
| :--- | :--- | :--- | :--- |
| **`West Bengal Female Reporter -1`** | Indian Bengali Female News Anchor | `West Bengal Female Reporter -1.wav` | `West Bengal Female Reporter -1.txt` |
| **`West Bengal Reporter 14 Sec`** | Indian Bengali News Anchor (14s sample) | `West Bengal Reporter 14 Sec.wav` | `West Bengal Reporter 14 Sec.txt` |
| **`Mujib`** | Historic 1952/1954/1958 speech | `Mujib.wav` | `Mujib.txt` |
| **`Bichar-Dilam-Final`** | Dramatic speech ("আল্লাহ তোমার কাছে বিচার দিলাম") | `Bichar-Dilam-Final.wav` | `Bichar-Dilam-Final.txt` |
| **`Bichar5`** | Dramatic quote sample | `Bichar5.wav` | `Bichar5.txt` |
| **`Sayed Johon short`** | Authentic clean YouTube host voice | `Sayed Johon short.wav` | `Sayed Johon short.txt` |
| **`Sayed_Johon_trimmed_7s`**| Ultra-fast 7s reference | `Sayed_Johon_trimmed_7s.wav` | `Sayed_Johon_trimmed_7s.txt` |
| **`Hasinacall`** | Cloned telephone reference | `Hasinacall.wav` | `Hasinacall.txt` |
| **`Sayed Johon A`** | Long narrative style | `Sayed Johon A.wav` | `Sayed Johon A.txt` |

---

## 🛠️ Features

- **🎙️ Zero-Shot Voice Cloning**: Clone any voice in seconds from reference audio + optional transcript.
- **🎨 Voice Design**: Generate voices from natural language descriptive prompts.
- **🎲 Auto Voice**: Generate unique random human vocal profiles.
- **📚 Voice Library Manager**: Upload new voices, edit reference transcripts, and persist them.
- **📜 History & Downloads**: Review, play back, and download all generated audio.

---

Built by [Sayed Johon](https://github.com/sayedjohon).
