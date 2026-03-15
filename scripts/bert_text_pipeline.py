"""
Depression Detection – Text Pipeline (BERT Fine-Tuning)
Dataset: DAIC-WOZ (Distress Analysis Interview Corpus – Wizard of Oz)
Model:   bert-base-uncased  |  10-fold cross-validation
Authors: Ayush Negi (22BLC1259), Ayush Arora (22BLC1218), Karan Ghuwalewala (22BLC1201)
VIT Chennai – School of Electronics Engineering, Nov 2025
"""

# ─────────────────────────────────────────────
# 0. Imports
# ─────────────────────────────────────────────
import os
import glob
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils import resample

# ─────────────────────────────────────────────
# 1. Configuration & Hyperparameters
# ─────────────────────────────────────────────
MODEL_NAME    = "bert-base-uncased"
TRANSCRIPT_DIR = "data/*.csv"          # path to DAIC-WOZ transcript CSVs
LABELS_CSV     = "data/labels.csv"     # columns: Participant_ID, PHQ_Binary
OUTPUT_CSV     = "data/balanced_data.csv"

MAX_LEN      = 128
BATCH_SIZE   = 8
EPOCHS       = 10
LEARNING_RATE = 2e-5
WEIGHT_DECAY  = 0.01
N_SPLITS     = 10
NUM_WORKERS  = 2
SEED         = 42

INTERROGATIVE_WORDS = (
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "is", "are", "am", "was", "were",
    "do", "does", "did",
    "has", "have", "had",
    "can", "could", "will", "would", "should", "may", "might",
    "tell me", "describe", "explain", "give me an example",
    "so,", "and what", "what about", "how about"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────────
# 2. Step 1 – Extract Q&A pairs from transcripts
# ─────────────────────────────────────────────
def extract_qa_pairs(transcript_dir: str) -> pd.DataFrame:
    all_qa = []
    transcript_files = glob.glob(transcript_dir)

    if not transcript_files:
        raise FileNotFoundError(f"No transcript files found at '{transcript_dir}'")

    for file_path in transcript_files:
        filename    = os.path.basename(file_path)
        person_id   = filename.split("_")[0]

        try:
            df = pd.read_csv(file_path, sep="\t", names=["Speaker", "value"], header=0)
            current_question  = ""
            current_answer_parts = []

            for _, row in df.iterrows():
                speaker  = str(row["Speaker"]).strip()
                dialogue = str(row["value"]).strip()

                if speaker == "Ellie":
                    if current_question and current_answer_parts:
                        all_qa.append({
                            "personId": person_id,
                            "question": current_question,
                            "answer":   " ".join(current_answer_parts)
                        })
                    current_question      = dialogue
                    current_answer_parts  = []

                elif speaker == "Participant":
                    if current_question and dialogue:
                        current_answer_parts.append(dialogue)

            # flush last pair
            if current_question and current_answer_parts:
                all_qa.append({
                    "personId": person_id,
                    "question": current_question,
                    "answer":   " ".join(current_answer_parts)
                })

        except Exception as e:
            print(f"Could not process file {file_path}: {e}")

    df_qa = pd.DataFrame(all_qa)
    df_qa.dropna(subset=["question", "answer"], inplace=True)
    df_qa = df_qa[df_qa["question"].str.strip() != ""]
    df_qa = df_qa[df_qa["answer"].str.strip() != ""]
    # keep only questions that start with an interrogative word
    df_qa = df_qa[df_qa["question"].str.lower().str.startswith(INTERROGATIVE_WORDS)]
    print(f"Total Q&A pairs extracted: {len(df_qa)}")
    return df_qa


# ─────────────────────────────────────────────
# 3. Step 2 – Merge with labels & balance dataset
# ─────────────────────────────────────────────
def merge_and_balance(df_qa: pd.DataFrame, labels_csv: str) -> pd.DataFrame:
    labels_df = pd.read_csv(labels_csv)
    labels_df["Participant_ID"] = labels_df["Participant_ID"].astype(int)
    df_qa["personId"]           = df_qa["personId"].astype(int)

    merged = df_qa.merge(labels_df, left_on="personId", right_on="Participant_ID", how="inner")
    merged.drop(columns=["Participant_ID"], inplace=True)

    majority = merged[merged["PHQ_Binary"] == 0]
    minority = merged[merged["PHQ_Binary"] == 1]

    print(f"Before balancing – Not Depressed: {len(majority)}, Depressed: {len(minority)}")

    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=SEED)
    balanced = pd.concat([majority, minority_upsampled]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    balanced["text"] = balanced["question"] + " [SEP] " + balanced["answer"]
    print(f"After balancing – total samples: {len(balanced)}")
    return balanced


# ─────────────────────────────────────────────
# 4. PyTorch Dataset
# ─────────────────────────────────────────────
class DepressionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "text":          self.texts[idx],
            "input_ids":     encoding["input_ids"].flatten(),
            "attention_mask":encoding["attention_mask"].flatten(),
            "labels":        torch.tensor(self.labels[idx], dtype=torch.long)
        }


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = DepressionDataset(
        texts   = df["text"].to_numpy(),
        labels  = df["PHQ_Binary"].to_numpy(),
        tokenizer=tokenizer,
        max_len =max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=NUM_WORKERS)


# ─────────────────────────────────────────────
# 5. Train & Evaluate helpers
# ─────────────────────────────────────────────
def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    losses = []
    for d in data_loader:
        input_ids      = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels         = d["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return np.mean(losses)


def eval_model(model, data_loader, device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for d in data_loader:
            input_ids      = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels         = d["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            probs   = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    predictions = (np.array(all_probs) >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(all_labels, predictions),
        "f1":       f1_score(all_labels, predictions),
        "roc_auc":  roc_auc_score(all_labels, all_probs),
        "labels":   all_labels,
        "preds":    predictions
    }


# ─────────────────────────────────────────────
# 6. Main – 10-fold cross-validation
# ─────────────────────────────────────────────
def run_kfold(df_full):
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    kfold     = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_f1, fold_roc, fold_acc = [], [], []
    master_labels, master_preds = [], []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(df_full)):
        print(f"\n──── Fold {fold + 1}/{N_SPLITS} ────")

        model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        model = model.to(device)

        df_train = df_full.iloc[train_ids]
        df_test  = df_full.iloc[test_ids]

        train_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
        test_loader  = create_data_loader(df_test,  tokenizer, MAX_LEN, BATCH_SIZE)

        optimizer     = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        total_steps   = len(train_loader) * EPOCHS
        scheduler     = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
            if train_loss is None:
                break
            print(f"  Epoch {epoch + 1}/{EPOCHS}  Training loss: {train_loss:.4f}")

        metrics = eval_model(model, test_loader, device)
        print(f"  Fold {fold + 1} | F1: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | Acc: {metrics['accuracy']:.4f}")

        fold_f1.append(metrics["f1"])
        fold_roc.append(metrics["roc_auc"])
        fold_acc.append(metrics["accuracy"])
        master_labels.extend(metrics["labels"])
        master_preds.extend(metrics["preds"])

    print("\n──── K-Fold Cross-Validation Complete ────")
    print(f"Average F1 Score: {np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f}")
    print(f"Average ROC-AUC:  {np.mean(fold_roc):.4f} ± {np.std(fold_roc):.4f}")
    print(f"Average Accuracy: {np.mean(fold_acc):.4f} ± {np.std(fold_acc):.4f}")

    cm = confusion_matrix(master_labels, master_preds)
    print("\n──── Aggregate Confusion Matrix ────")
    print(f"  TN: {cm[0,0]}  FP: {cm[0,1]}")
    print(f"  FN: {cm[1,0]}  TP: {cm[1,1]}")


# ─────────────────────────────────────────────
# 7. Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df_qa      = extract_qa_pairs(TRANSCRIPT_DIR)
    df_balanced = merge_and_balance(df_qa, LABELS_CSV)
    df_balanced.to_csv(OUTPUT_CSV, index=False)
    print(f"Balanced data saved to '{OUTPUT_CSV}'")
    run_kfold(df_balanced)
