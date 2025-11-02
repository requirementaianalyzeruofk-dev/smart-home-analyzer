
# finetuned_model.py
# Final — Smart Home Access System (2 classes), 1000 diverse samples
# Improvements: weight_decay, warmup_ratio, EarlyStopping, load_best_model_at_end
# epochs = 5, prints single strongest label + confidence

import random, re, os
import numpy as np
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
# Utilities
# ------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s\.,\-\/\:\(\)]", "", text)
    return text

# ------------------------------
# Generators for Smart Home Access System (diverse)
# ------------------------------
openers = ["The system shall", "The application must", "The product must", "The smart lock system shall", "The platform must"]
doors = ["front door", "garage door", "main entrance", "patio door", "bedroom door"]
auth_methods = ["biometric fingerprint", "face recognition", "PIN code", "password + 2FA", "NFC tag", "RFID card"]
guest_flows = ["temporary guest code", "time-limited access token", "one-time access link", "scheduled entry permit"]
events = ["entry", "exit", "failed authentication", "tamper alert", "forced entry", "power failure", "network loss"]
metrics = ["response time", "uptime", "latency", "throughput", "success rate"]
values = ["1", "2", "3", "5", "10"]
time_units = ["seconds", "milliseconds", "minutes"]
security_measures = ["AES-256", "TLS 1.3", "multi-factor authentication", "SHA-512"]
scenarios = [
    "during power failure", "under network outage", "when offline", "during high load",
    "when multiple users attempt concurrent access", "during firmware update", "in emergency mode"
]

def make_functional():
    templates = [
        "{opener} allow an authorized user to unlock the {door} via {auth}.",
        "{opener} issue a {guest} to a visitor when authorized by an administrator.",
        "{opener} revoke access for any user instantly via admin console.",
        "{opener} log every {event} with timestamps for auditing.",
        "{opener} provide a mobile interface to export access logs.",
        "{opener} allow scheduling of automatic locks at defined times.",
        "{opener} support integration with smart cameras and voice assistants.",
        "{opener} allow backup physical key access in case of system failure.",
        "{opener} allow administrators to assign time-based access privileges."
    ]
    tpl = random.choice(templates)
    return clean_text(tpl.format(
        opener=random.choice(openers),
        door=random.choice(doors),
        auth=random.choice(auth_methods),
        guest=random.choice(guest_flows),
        event=random.choice(events)
    ))

def make_nonfunc():
    templates = [
        "{opener} maintain {metric} below {value} {time_unit} under typical load.",
        "{opener} ensure data is encrypted using {sec} both in transit and at rest.",
        "{opener} maintain {value}% uptime excluding scheduled maintenance.",
        "{opener} handle up to {value} concurrent authenticated sessions without failure.",
        "{opener} recover to operational state within {value} {time_unit} after power restoral.",
        "{opener} comply with privacy regulations and keep audit logs for at least {value} days.",
        "{opener} keep average {metric} under {value} {time_unit} even {scenario}.",

"{opener} provide automatic secure firmware updates without downtime."
    ]
    tpl = random.choice(templates)
    return clean_text(tpl.format(
        opener=random.choice(openers),
        metric=random.choice(metrics),
        value=random.choice(values),
        time_unit=random.choice(time_units),
        sec=random.choice(security_measures),
        scenario=random.choice(scenarios)
    ))

# ------------------------------
# Build dataset: 1000 balanced (Functional=0, Non-Functional=1)
# ------------------------------
TOTAL = 1000
half = TOTAL // 2

data = []
# small set of hand-crafted seeds to anchor linguistic patterns (optional)
seed_examples = [
    {"requirement_text": "The system shall unlock doors via the mobile application.", "label": 0},
    {"requirement_text": "Users must be able to create temporary access codes for guests.", "label": 0},
    {"requirement_text": "All user data shall be encrypted using AES-256 encryption.", "label": 1},
    {"requirement_text": "The system must respond to user commands within 2 seconds under normal load.", "label": 1},
]
for s in seed_examples:
    data.append({"requirement_text": clean_text(s["requirement_text"]), "label": s["label"]})

# generate until balanced counts reached
while len([d for d in data if d["label"] == 0]) < half:
    data.append({"requirement_text": make_functional(), "label": 0})
while len([d for d in data if d["label"] == 1]) < half:
    data.append({"requirement_text": make_nonfunc(), "label": 1})

random.shuffle(data)
print(f"✅ Generated dataset size: {len(data)} (Functional={sum(1 for d in data if d['label']==0)}, NonFunctional={sum(1 for d in data if d['label']==1)})")

# ------------------------------
# Prepare HuggingFace Dataset
# ------------------------------
hf_dataset = Dataset.from_list([{"requirement_text": d["requirement_text"], "label": d["label"]} for d in data])
split = hf_dataset.train_test_split(test_size=0.12, seed=RANDOM_SEED)
train_dataset = split["train"]
eval_dataset = split["test"]

# Tokenizer & model
MODEL_NAME = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def tokenize_fn(batch):
    return tokenizer(batch["requirement_text"], padding="max_length", truncation=True, max_length=128)

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
# Training arguments (with improvements)
# ------------------------------
output_dir = "./finetuned_roberta_binary"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,          # stabilizes training
    warmup_ratio=0.1,           # warmup for first 10% steps
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_binary",
    greater_is_better=True,
    logging_steps=50,
    fp16=False,                 # set True only if your GPU supports it
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
print("🚀 Starting RoBERTa fine-tuning (Smart Home Access, 2 classes)...")
trainer.train()
print("✅ Training finished.")

# ------------------------------
# Save model & tokenizer
# ------------------------------
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ Model and tokenizer saved to: {output_dir}")

# ------------------------------
# Inference helper (single strongest label only)
# ------------------------------
classifier = pipeline("text-classification", model=output_dir, tokenizer=output_dir, device=0 if DEVICE_IS_GPU else -1)

label_map = {0: "Functional", 1: "Non-Functional"}

def predict_one(text: str):
    res = classifier(text)[0]
    label_id = int(res["label"].replace("LABEL_", ""))
    score = res["score"]
    return label_map[label_id], score * 100.0

# quick examples
examples = [
    "The system shall unlock the front door via biometric fingerprint.",
    "The system must maintain response time under 2 seconds.",
    "In case of network loss, the system shall allow offline access via backup key."
]
print("\n🔍 Example predictions (single strongest label):")
for t in examples:
    lbl, cf = predict_one(t)
    print(f"- \"{t}\" → {lbl} (confidence={cf:.2f}%)")

# Interactive: allow user to type requirements and get prediction
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