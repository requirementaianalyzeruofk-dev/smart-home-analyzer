# finetuned_model_promise.py
# Fine-tuning RoBERTa on PROMISE dataset (Functional vs Non-Functional)
# CPU-compatible version

import re, os, random
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback, pipeline
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

# ------------------------------
# Device check
# ------------------------------
print("🔥 Running on:", "GPU ✅" if torch.cuda.is_available() else "CPU 💻")
DEVICE_IS_GPU = torch.cuda.is_available()

# ------------------------------
# Reproducibility
# ------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ------------------------------
# Text cleaning
# ------------------------------
def clean_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s\.,\-\/\:\(\)]", "", text)
    return text

# ------------------------------
# Load PROMISE dataset
# ------------------------------
csv_path = "promise_data.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ PROMISE CSV file not found: {csv_path}")

df = pd.read_csv(csv_path)

# Ensure column names exist
if not {"RequirementText", "IsFunctional"}.issubset(df.columns):
    raise ValueError(f"❌ Expected columns 'RequirementText' and 'IsFunctional' but got {list(df.columns)}")

# Rename for convenience
df = df.rename(columns={"RequirementText": "requirement", "IsFunctional": "label"})

# Drop missing rows
df.dropna(subset=["requirement", "label"], inplace=True)

# Convert label: 1 → Functional, 0 → Non-Functional (if needed)
# Adjust depending on your data labeling convention
df["label"] = df["label"].astype(int)

df["requirement"] = df["requirement"].apply(clean_text)

print(f"✅ Loaded PROMISE dataset: {len(df)} samples (Functional={sum(df['label']==1)}, Non-Functional={sum(df['label']==0)})")

# ------------------------------
# Prepare HuggingFace Dataset
# ------------------------------
hf_dataset = Dataset.from_pandas(df[["requirement", "label"]])
split = hf_dataset.train_test_split(test_size=0.2, seed=RANDOM_SEED)
train_dataset, eval_dataset = split["train"], split["test"]

# ------------------------------
# Tokenizer & Model
# ------------------------------
MODEL_NAME = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def tokenize_fn(batch):
    return tokenizer(batch["requirement"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_fn, batched=True)
eval_dataset  = eval_dataset.map(tokenize_fn, batched=True)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ------------------------------
# Metrics
# ------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_binary": f1_score(labels, preds, average="binary")
    }

# ------------------------------
# Training arguments
# ------------------------------
output_dir = "./finetuned_roberta_promise"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_binary",
    greater_is_better=True,
    logging_steps=50,
    fp16=False,  # CPU safe
    save_total_limit=2,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# ------------------------------
# Start training
# ------------------------------
print("🚀 Starting RoBERTa fine-tuning on PROMISE dataset (Functional vs Non-Functional)...")
trainer.train()
print("✅ Training finished successfully.")

# ------------------------------
# Save model & tokenizer
# ------------------------------
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ Model and tokenizer saved to: {output_dir}")

# ------------------------------
# Inference helper
# ------------------------------
classifier = pipeline("text-classification", model=output_dir, tokenizer=output_dir, device=0 if DEVICE_IS_GPU else -1)
label_map = {0: "Non-Functional", 1: "Functional"}

def predict_one(text: str):
    res = classifier(text)[0]
    label_id = int(res["label"].replace("LABEL_", ""))
    score = res["score"]
    return label_map[label_id], score * 100.0

# ------------------------------
# Quick test examples
# ------------------------------
examples = [
    "The system shall allow users to register new accounts.",
    "The system must maintain a response time under 2 seconds.",
    "All data shall be encrypted using AES-256 encryption."
]
print("\n🔍 Example predictions:")
for t in examples:
    lbl, cf = predict_one(t)
    print(f"- \"{t}\" → {lbl} (confidence={cf:.2f}%)")

# ------------------------------
# Interactive mode
# ------------------------------
print("\nType a requirement (or 'exit' to quit):")
while True:
    try:
        user_text = input("> ").strip()
        if user_text.lower() in ("exit", "quit"):
            print("Goodbye.")
            break
        lbl, cf = predict_one(user_text)
        print(f"→ {lbl}  |  Confidence: {cf:.2f}%\n")
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
        break
