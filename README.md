<div align="center">

# 🧠 Deep Mental Health Voice
### Depression Detection from Speech & Language using Transformers

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21F?logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green)

**B.Tech Final Year Project · School of Electronics Engineering · VIT Chennai · Nov 2025**

Ayush Negi (22BLC1259) · Ayush Arora (22BLC1218) · Karan Ghuwalewala (22BLC1201)

*Supervisor: Dr. Ashok Mondal*

</div>

---

## 💡 What is this?

Depression affects over **332 million people** worldwide, yet most diagnostic methods rely on face-to-face interviews and patient self-disclosure — creating barriers around stigma, access, and awareness.

This project asks a different question: **can we detect depression from the way someone speaks and what they say?**

We train two transformer-based deep learning models on the **DAIC-WOZ clinical dataset** — one analyzing *language patterns* in interview transcripts, and one analyzing *vocal patterns* in audio recordings — without ever directly asking the patient about their mental state.

---

## 📊 Results at a Glance

| Modality | Model | Accuracy | F1-Score | ROC-AUC |
|----------|-------|:--------:|:--------:|:-------:|
| 📝 Text (transcripts) | BERT `bert-base-uncased` | **76.33%** | **0.78** | **0.84** |
| 🎙️ Audio (Mel-spectrograms) | ViT `vit_base_patch16_224` | **69.0%** | — | — |

---

## 🗂️ Repository Structure

```
deep-mental-health-voice/
│
├── 📓 notebooks/
│   ├── 01_BERT_Text_Pipeline.ipynb      ← Colab-ready, run top to bottom
│   └── 02_ViT_Audio_Pipeline.ipynb      ← Colab-ready, run top to bottom
│
├── 🐍 scripts/
│   ├── bert_text_pipeline.py            ← Standalone BERT script
│   └── vit_audio_pipeline.py            ← Standalone ViT script
│
├── data/                                ← Not committed (add your own)
│   ├── AUDIO/                           Raw .wav interview recordings
│   ├── TRANSCRIPTS/                     Transcript .csv files
│   ├── csvs/                            Label files (PHQ_Binary)
│   ├── Cleaned_Audio/                   Participant-only segments
│   ├── Denoised_Audio/                  Noise-reduced audio
│   ├── Spectrograms_Optimized/          Generated Mel-spectrogram PNGs
│   └── ViT_Results/                     Saved model weights + embeddings
│
└── README.md
```

---

## 🔬 Methodology

### 📝 Text Pipeline — BERT

```
Transcripts → Q&A Pair Extraction → Random Oversampling
    → BERT Tokenizer (max_len=128) → Fine-tune bert-base-uncased
        → 10-Fold Cross Validation → Depressed / Not Depressed
```

- Interview transcripts are split into **question-answer pairs** using interrogative word detection
- Class imbalance handled via **random oversampling** (2133 samples each class)
- Fine-tuned with **AdamW** (lr=2e-5, 10 epochs, batch size 8)
- Validated with **10-fold cross-validation** for stable metrics

### 🎙️ Audio Pipeline — Vision Transformer

```
Raw .wav → Extract Participant Segments → Noise Reduction
    → Mel-Spectrogram (PNG) → ViT-Base Patch16 Fine-tuning
        → Depressed / Not Depressed
```

- Only **participant speech segments** are extracted (Ellie's voice removed)
- Audio denoised using `noisereduce` with first 0.5s as noise profile
- Converted to **Mel-spectrograms** (64 mel bins, magma colormap, 224×224)
- Class-weighted **CrossEntropyLoss** + early stopping (patience=6)
- Data split: 70% train / 15% val / 15% test

---

## 📈 Comparison with Prior Work

### Text Modality

| Study | Model | F1-Score |
|-------|-------|:--------:|
| Petrov et al. | RoBERTa (Hierarchical Regression) | 0.739 |
| Chi et al. | RoBERTa (Participant only) | 0.750 |
| Mustafa et al. | DepRoBERTa | 0.710 |
| Zhang et al. | Multi-MT5 + Multi-RoBERTa | 0.880 |
| **This work** | **BERT** | **0.782** |

### Audio Modality

| Study | Model | Accuracy |
|-------|-------|:--------:|
| Zhou et al. | CNN (Handcrafted features) | 83% |
| Milo et al. | CNN (Baseline) | 82.5% |
| Feng et al. | CNN (Acoustic features) | 77% |
| Ravi et al. | DepAudioNet | 74% |
| **This work** | **ViT** | **69%** |

---

## ⚙️ Setup & Usage

### Requirements

```bash
pip install torch transformers timm librosa noisereduce pydub soundfile \
            scikit-learn pandas numpy matplotlib seaborn
apt-get install -y ffmpeg   # for audio processing
```

### 🚀 Run on Google Colab (recommended)

1. Get DAIC-WOZ dataset access at [dcapswoz.ict.usc.edu](https://dcapswoz.ict.usc.edu/)
2. Upload to Google Drive under `MyDrive/DDS/`
3. Open a notebook from `notebooks/` in Colab and run all cells

### 💻 Run Locally

```bash
# Text pipeline
python scripts/bert_text_pipeline.py

# Audio pipeline
python scripts/vit_audio_pipeline.py
```

> Edit the path constants at the top of each script to match your local data directory.

---

## 🔑 Key Findings

- **BERT significantly outperforms ViT** for single-modality depression detection on DAIC-WOZ
- Linguistic cues — negative word usage, self-referential language, flat affect in responses — are stronger signals than audio features alone when using ViT
- ViT on Mel-spectrograms is a novel approach but traditional CNN-based pipelines currently lead on audio tasks
- **Multimodal fusion** (combining BERT text embeddings + ViT audio embeddings) is the most promising direction for future work

---

## 🔭 Future Work

- [ ] **Multimodal early fusion** — combine BERT + ViT embeddings for a unified classifier
- [ ] **Real-time app** — deploy as a smartphone or web interface for passive screening
- [ ] **Explainability (XAI)** — highlight which words/audio segments drove the prediction
- [ ] **Multilingual expansion** — extend to Hindi, Tamil, and other regional languages

---

## 📚 Dataset

**DAIC-WOZ** — Distress Analysis Interview Corpus (Wizard of Oz)  
189 clinical interviews between participants and a virtual agent "Ellie".  
Labels based on **PHQ-8** clinical questionnaire scores (≥ 10 = Depressed).

> ⚠️ Dataset access requires approval from USC ICT: https://dcapswoz.ict.usc.edu/

---

## 📄 References

1. Zhang et al. (2025) – Multi-instance learning with Multi-MT5/RoBERTa. *Sci Rep.*
2. Milintsevich et al. (2023) – Hierarchical RoBERTa regression. *Brain Informatics.*
3. Burdisso et al. (2024) – RoBERTa on participant responses only. *ClinicalNLP Workshop.*
4. Triohin et al. (2025) – CNN-to-SNN for audio depression detection. *Applied Sciences.*
5. Nykoniuk et al. (2025) – Multimodal fusion with CNN + BiLSTM. *Computation.*

Full reference list in the project report PDF.
