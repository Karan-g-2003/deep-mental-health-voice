# Depression Detection using Deep Learning

> **B.Tech Project Report** – School of Electronics Engineering, VIT Chennai (Nov 2025)
> 
> **Authors:** Ayush Negi (22BLC1259) · Ayush Arora (22BLC1218) · Karan Ghuwalewala (22BLC1201)
> 
> **Supervisor:** Dr. Ashok Mondal

---

## Overview

This project explores automated depression detection using two modalities — **speech** and **text** — from clinical interview recordings. Instead of relying on explicit self-disclosure, the system learns subtle linguistic and acoustic patterns that correlate with depression.

| Modality | Model | Accuracy | F1-Score |
|----------|-------|----------|----------|
| Text (transcripts) | BERT (`bert-base-uncased`) | **76.33%** | **0.78** |
| Audio (Mel-spectrograms) | Vision Transformer (`ViT-Base Patch16`) | **69%** | – |

---

## Dataset

**DAIC-WOZ** (Distress Analysis Interview Corpus – Wizard of Oz)  
Participants interact with a virtual agent "Ellie" in a structured clinical interview. Labels are derived from **PHQ-8** scores (≥ 10 = Depressed).

- 189 interviews
- Modalities used: Audio (`.wav`) + Text transcripts (`.csv`)
- Labels: `0` = Not Depressed, `1` = Depressed

> The dataset requires approval from USC ICT. See: https://dcapswoz.ict.usc.edu/

---

## Repository Structure

```
depression-detection/
│
├── notebooks/
│   ├── 01_BERT_Text_Pipeline.ipynb       # Full BERT pipeline (Colab-ready)
│   └── 02_ViT_Audio_Pipeline.ipynb       # Full ViT pipeline (Colab-ready)
│
├── scripts/
│   ├── bert_text_pipeline.py             # Standalone BERT script
│   └── vit_audio_pipeline.py             # Standalone ViT script
│
├── data/                                 # (not committed – add your own)
│   ├── AUDIO/                            # Raw .wav interview files
│   ├── TRANSCRIPTS/                      # Transcript .csv files
│   ├── csvs/                             # Label CSVs (PHQ_Binary)
│   ├── Cleaned_Audio/                    # Participant-only audio segments
│   ├── Denoised_Audio/                   # Noise-reduced audio
│   ├── Spectrograms_Optimized/           # Generated Mel-spectrogram PNGs
│   └── ViT_Results/                      # Saved model + embeddings
│
└── README.md
```

---

## Methodology

### Text Pipeline (BERT)

1. Extract **Q&A pairs** from transcripts using interrogative word detection
2. Balance classes via **random oversampling** (2133 each)
3. Tokenize with `bert-base-uncased` tokenizer (max length = 128)
4. Fine-tune BERT with **AdamW** optimizer (lr = 2e-5, 10 epochs)
5. Evaluate with **10-fold cross-validation**

### Audio Pipeline (ViT)

1. Extract **participant-only** audio segments using transcript timestamps
2. **Denoise** using `noisereduce` (first 0.5s as noise profile)
3. Convert to **Mel-spectrograms** (64 mel bins, magma colormap) via `librosa`
4. Resize to 224×224 and fine-tune **ViT-Base Patch16** with class-weighted loss
5. Train/val/test split: 70% / 15% / 15%

---

## Results

### BERT (Text)
- Average Accuracy: **0.7633 ± 0.0183**
- Average F1:       **0.7817 ± 0.0150**
- Average ROC-AUC:  **0.8375 ± 0.0173**

### ViT (Audio)
- Final Accuracy: **69.0%** (Best Val Acc during training: 69.0%)

### Comparison with Prior Work (Text Modality)

| Study | Model | F1-Score |
|-------|-------|----------|
| Petrov et al. | RoBERTa (Hierarchical) | 0.739 |
| Chi et al. | RoBERTa (Participant only) | 0.75 |
| Mustafa et al. | DepRoBERTa | 0.71 |
| Zhang et al. | Multi-MT5 + Multi-RoBERTa | 0.88 |
| **This work** | **BERT** | **0.78** |

---

## Setup & Usage

### Requirements

```bash
pip install torch transformers timm librosa noisereduce pydub soundfile \
            scikit-learn pandas numpy matplotlib seaborn
# For audio processing
apt-get install -y ffmpeg
```

### Run on Google Colab (recommended)

1. Upload dataset to Google Drive under `MyDrive/DDS/`
2. Open either notebook in `notebooks/` via Colab
3. Run all cells top to bottom

### Run as a Script (local)

```bash
# Text pipeline
python scripts/bert_text_pipeline.py

# Audio pipeline
python scripts/vit_audio_pipeline.py
```

> Update the path constants at the top of each script to point to your local data.

---

## Key Findings

- **BERT outperforms ViT** for single-modality depression detection on this dataset
- Text-based linguistic cues (negative sentiment, self-referential language) are stronger signals than audio alone with ViT
- ViT applied to Mel-spectrograms is promising but traditional CNNs currently outperform it for audio-based depression detection
- A **multimodal fusion** of BERT + ViT embeddings is identified as the most promising future direction

---

## Future Work

- Multimodal early fusion (BERT text embeddings + ViT audio embeddings)
- Real-time deployment as a smartphone / web app
- Explainability (XAI) — show *why* a prediction was made
- Multilingual dataset expansion (Hindi, Tamil, regional accents)

---

## References

Key references from the literature survey:

1. Zhang et al. (2025) – Multi-instance learning with Multi-MT5/RoBERTa. *Sci Rep.*
2. Milintsevich et al. (2023) – Hierarchical RoBERTa regression. *Brain Informatics.*
3. Burdisso et al. (2024) – RoBERTa on participant-only responses. *ClinicalNLP Workshop.*
4. Triohin et al. (2025) – CNN-to-SNN conversion for audio depression detection. *Applied Sciences.*
5. Nykoniuk et al. (2025) – Multimodal fusion (CNN + BiLSTM). *Computation.*

Full reference list available in the project report.
