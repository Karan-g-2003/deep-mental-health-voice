"""
Depression Detection – Audio Pipeline (ViT Spectrogram Classification)
Dataset: DAIC-WOZ (Distress Analysis Interview Corpus – Wizard of Oz)
Model:   vit_base_patch16_224  |  Fine-tuned on Mel-Spectrograms
Authors: Ayush Negi (22BLC1259), Ayush Arora (22BLC1218), Karan Ghuwalewala (22BLC1201)
VIT Chennai – School of Electronics Engineering, Nov 2025
"""

# ─────────────────────────────────────────────
# 0. Imports
# ─────────────────────────────────────────────
import os
import re
import gc
import copy
import time
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import librosa
import librosa.display
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# ─────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────
BASE_DIR         = "data"
AUDIO_DIR        = os.path.join(BASE_DIR, "AUDIO")
TRANSCRIPT_DIR   = os.path.join(BASE_DIR, "TRANSCRIPTS")
CLEANED_AUDIO    = os.path.join(BASE_DIR, "Cleaned_Audio")
DENOISED_AUDIO   = os.path.join(BASE_DIR, "Denoised_Audio")
SPECTROGRAM_DIR  = os.path.join(BASE_DIR, "Spectrograms_Optimized")
RESULTS_DIR      = os.path.join(BASE_DIR, "ViT_Results")
CSV_DIR          = os.path.join(BASE_DIR, "csvs")

for d in [CLEANED_AUDIO, DENOISED_AUDIO, SPECTROGRAM_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

IMG_SIZE     = 224
BATCH_SIZE   = 8
EPOCHS       = 25
LR           = 1e-4
NUM_WORKERS  = 2
PATIENCE     = 6
MODEL_NAME   = "vit_base_patch16_224"
SEED         = 42

BEST_MODEL_PATH  = os.path.join(RESULTS_DIR, "best_vit.pth")
EMBEDDING_FILE   = os.path.join(RESULTS_DIR, "embeddings.npy")
EMBEDDING_CSV    = os.path.join(RESULTS_DIR, "embeddings_index.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {device}")


# ─────────────────────────────────────────────
# 2. Step 1 – Extract participant-only audio segments
# ─────────────────────────────────────────────
def extract_id(filename: str):
    m = re.match(r"(\d+)", filename)
    return m.group(1) if m else None


def extract_participant_audio():
    """
    For each transcript CSV, extract only the participant's speaking segments
    from the matching WAV file and save as a clean WAV.
    """
    transcript_files = [f for f in os.listdir(TRANSCRIPT_DIR) if f.endswith(".csv")]
    print(f"Found {len(transcript_files)} transcript files.")

    for filename in transcript_files:
        transcript_id = extract_id(filename)
        if not transcript_id:
            print(f"  Skipping {filename}: could not extract ID.")
            continue

        matching_audio = [a for a in os.listdir(AUDIO_DIR)
                          if transcript_id in a and a.endswith(".wav")]
        if not matching_audio:
            print(f"  No audio found for {filename}")
            continue

        wav_path  = os.path.join(AUDIO_DIR,        matching_audio[0])
        csv_path  = os.path.join(TRANSCRIPT_DIR,   filename)
        out_path  = os.path.join(CLEANED_AUDIO,    f"{transcript_id}_Participant.wav")

        if os.path.exists(out_path):
            continue  # already processed

        df = pd.read_csv(csv_path)
        participant_df = df[df["speaker"] == "Participant"]

        audio = AudioSegment.from_wav(wav_path)
        participant_audio = AudioSegment.empty()
        for _, row in participant_df.iterrows():
            start_ms = row["start_time"] * 1000
            end_ms   = row["stop_time"]  * 1000
            participant_audio += audio[start_ms:end_ms]

        participant_audio.export(out_path, format="wav")
        print(f"  Saved: {out_path}")

    print("Participant audio extraction complete.")


# ─────────────────────────────────────────────
# 3. Step 2 – Denoise audio
# ─────────────────────────────────────────────
def denoise_audio():
    """Apply noise reduction to every cleaned WAV file."""
    wav_files = [f for f in os.listdir(CLEANED_AUDIO) if f.endswith(".wav")]
    print(f"Denoising {len(wav_files)} files...")

    for filename in wav_files:
        input_path  = os.path.join(CLEANED_AUDIO,  filename)
        output_path = os.path.join(DENOISED_AUDIO, filename.replace(".wav", "_denoised.wav"))

        if os.path.exists(output_path):
            continue

        y, sr = librosa.load(input_path, sr=None)
        noise_sample   = y[:int(0.5 * sr)]          # first 0.5 s as noise profile
        reduced_audio  = nr.reduce_noise(y=y, y_noise=noise_sample, sr=sr, prop_decrease=0.95)
        sf.write(output_path, reduced_audio, sr)
        print(f"  Saved: {output_path}")

    print("All files denoised successfully.")


# ─────────────────────────────────────────────
# 4. Step 3 – Generate Mel-spectrograms
# ─────────────────────────────────────────────
def generate_spectrograms():
    """Convert each denoised WAV into a Mel-spectrogram PNG."""
    wav_files  = [f for f in os.listdir(DENOISED_AUDIO) if f.endswith(".wav")]
    batch_size = 10
    print(f"Generating spectrograms for {len(wav_files)} files in batches of {batch_size}...")

    for i in range(0, len(wav_files), batch_size):
        batch = wav_files[i:i + batch_size]
        print(f"  Processing batch {i // batch_size + 1}/{len(wav_files) // batch_size + 1}")

        for filename in batch:
            file_path  = os.path.join(DENOISED_AUDIO,  filename)
            out_path   = os.path.join(SPECTROGRAM_DIR, filename.replace(".wav", "_spec.png"))

            if os.path.exists(out_path):
                continue

            try:
                y, sr = librosa.load(file_path, sr=None)
                S     = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
                S_dB  = librosa.power_to_db(S, ref=np.max)

                plt.figure(figsize=(8, 3))
                librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", cmap="magma")
                plt.axis("off")
                plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
                plt.close("all")
                print(f"    Saved {filename}")
            except Exception as e:
                print(f"    Skipped {filename}: {e}")

        gc.collect()

    print("All spectrograms generated.")


# ─────────────────────────────────────────────
# 5. Step 4 – Load labels & match spectrograms
# ─────────────────────────────────────────────
def load_labeled_spectrograms():
    csv_files = glob.glob(os.path.join(CSV_DIR, "*.csv")) + glob.glob(os.path.join(CSV_DIR, "*.xls*"))
    dfs = []
    for path in csv_files:
        try:
            dfs.append(pd.read_csv(path))
        except Exception:
            dfs.append(pd.read_excel(path))

    labels_df = (
        pd.concat(dfs, ignore_index=True)[["Participant_ID", "PHQ_Binary"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    labels_df["Participant_ID"] = labels_df["Participant_ID"].astype(str)

    spec_files = sorted([f for f in os.listdir(SPECTROGRAM_DIR) if f.lower().endswith(".png")])
    rows = [{"filename": f, "Participant_ID": extract_id(f)} for f in spec_files]
    files_df = pd.DataFrame(rows)

    merged = files_df.merge(labels_df, on="Participant_ID", how="left")
    merged = merged.dropna(subset=["PHQ_Binary"]).reset_index(drop=True)
    merged["label"] = merged["PHQ_Binary"].astype(int)

    print(f"Labeled spectrograms: {len(merged)}")
    print(merged["label"].value_counts())
    return merged


# ─────────────────────────────────────────────
# 6. PyTorch Dataset
# ─────────────────────────────────────────────
class SpectrogramDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.root      = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.root, row["filename"])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(row["label"]), row["filename"]


def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tf, val_tf


# ─────────────────────────────────────────────
# 7. Build ViT model
# ─────────────────────────────────────────────
def build_vit_model():
    model   = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    in_ch   = model.num_features
    model.head = nn.Sequential(
        nn.Linear(in_ch, 256),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    return model.to(device)


# ─────────────────────────────────────────────
# 8. Training loop
# ─────────────────────────────────────────────
def train_vit(merged: pd.DataFrame):
    train_df, val_df = train_test_split(
        merged, test_size=0.2, stratify=merged["label"], random_state=SEED
    )

    train_tf, val_tf = get_transforms()
    train_loader = DataLoader(
        SpectrogramDataset(train_df, SPECTROGRAM_DIR, train_tf),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        SpectrogramDataset(val_df, SPECTROGRAM_DIR, val_tf),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    model = build_vit_model()

    # class-weighted loss to handle imbalance
    labels_all    = merged["label"].values
    class_counts  = np.bincount(labels_all)
    weights       = torch.tensor(
        [len(labels_all) / (2 * c) for c in class_counts], dtype=torch.float32
    ).to(device)
    criterion  = nn.CrossEntropyLoss(weight=weights)
    optimizer  = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    from torch.cuda.amp import GradScaler, autocast
    scaler = GradScaler()

    best_acc, best_state, no_improve = 0.0, None, 0
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_preds, train_labels = 0.0, [], []

        for imgs, lbls, _ in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            with autocast():
                out  = model(imgs)
                loss = criterion(out, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss  += loss.item() * imgs.size(0)
            train_preds.extend(out.argmax(1).cpu().tolist())
            train_labels.extend(lbls.cpu().tolist())

        train_loss /= len(train_loader.dataset)
        train_acc   = accuracy_score(train_labels, train_preds)

        # Validation
        model.eval()
        val_loss, val_preds, val_labels = 0.0, [], []
        with torch.no_grad():
            for imgs, lbls, _ in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                with autocast():
                    out  = model(imgs)
                    loss = criterion(out, lbls)
                val_loss  += loss.item() * imgs.size(0)
                val_preds.extend(out.argmax(1).cpu().tolist())
                val_labels.extend(lbls.cpu().tolist())

        val_loss /= len(val_loader.dataset)
        val_acc   = accuracy_score(val_labels, val_preds)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d}: Train {train_acc:.3f} | Val {val_acc:.3f} | Loss {val_loss:.4f}")

        if val_acc > best_acc:
            best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())
            torch.save(best_state, BEST_MODEL_PATH)
            print(f"  ✅ Best model saved (Val Acc: {best_acc:.3f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("  ⏹ Early stopping triggered.")
                break

    print(f"Training done in {(time.time() - start) / 60:.1f} min. Best Val Acc={best_acc:.3f}")

    # Final evaluation
    if best_state:
        model.load_state_dict(best_state)
    all_loader = DataLoader(
        SpectrogramDataset(merged, SPECTROGRAM_DIR, val_tf),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    model.eval()
    y_true, y_pred, embeddings, meta = [], [], [], []
    with torch.no_grad():
        for imgs, lbls, fnames in all_loader:
            imgs = imgs.to(device)
            with autocast():
                out   = model(imgs)
                feats = model.forward_features(imgs)
                if isinstance(feats, tuple):
                    feats = feats[0]
            y_pred.extend(out.argmax(1).cpu().tolist())
            y_true.extend(lbls.tolist())
            embeddings.append(feats.cpu().numpy())
            meta += [{"filename": f, "label": int(l)} for f, l in zip(fnames, lbls)]

    emb_array = np.concatenate(embeddings)
    np.save(EMBEDDING_FILE, emb_array)
    pd.DataFrame(meta).to_csv(EMBEDDING_CSV, index=False)

    final_acc = accuracy_score(y_true, y_pred)
    print(f"Final overall accuracy: {final_acc}")
    print(f"All results stored in: {RESULTS_DIR}")

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]}  FP: {cm[0,1]}")
    print(f"  FN: {cm[1,0]}  TP: {cm[1,1]}")


# ─────────────────────────────────────────────
# 9. Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Step 1: Extract participant audio ===")
    extract_participant_audio()

    print("\n=== Step 2: Denoise audio ===")
    denoise_audio()

    print("\n=== Step 3: Generate Mel-spectrograms ===")
    generate_spectrograms()

    print("\n=== Step 4: Train Vision Transformer ===")
    merged = load_labeled_spectrograms()
    train_vit(merged)
