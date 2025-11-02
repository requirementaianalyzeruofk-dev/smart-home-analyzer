
# finetuned_model.py
# FINAL VERSION (CLEAN): RoBERTa with Simplified 2-Class Classification
# Functional (0) vs Non-Functional/Ambiguous (1)
# Includes Full Data Cleaning 🧹✅

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
from datasets import Dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import random
import re

# ------------------------------
# Reproducibility
# ------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ------------------------------
# 0) Text Cleaning Function 🧽
# ------------------------------
def clean_text(text):
    """تنظيف النص من الرموز الغريبة والمسافات الزائدة وتوحيد التنسيق"""
    text = text.strip()  # إزالة المسافات الزائدة في البداية والنهاية
    text = re.sub(r'\s+', ' ', text)  # توحيد المسافات
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\-\%]', '', text)  # إزالة الرموز الغريبة
    text = text.replace('  ', ' ')
    return text

# ------------------------------
# 1) Seed Data (Functional vs Non-Functional/Ambiguous)
# ------------------------------
seed_data = [
    # Functional (Label 0)
    {"requirement_text": "The system shall unlock doors via the mobile application.", "label": 0},
    {"requirement_text": "Users must be able to create temporary access codes for guests.", "label": 0},
    {"requirement_text": "The system shall log all entry and exit events.", "label": 0},
    {"requirement_text": "Administrators can add, remove, or modify user accounts.", "label": 0},
    {"requirement_text": "The system shall send notifications when unauthorized access is detected.", "label": 0},
    {"requirement_text": "The system shall allow remote locking and unlocking of doors.", "label": 0},
    {"requirement_text": "Users must be able to integrate the access system with smart cameras.", "label": 0},
    {"requirement_text": "The system shall allow scheduling of automatic door locks.", "label": 0},
    {"requirement_text": "The system shall allow multiple users to share access privileges.", "label": 0},
    {"requirement_text": "The system shall provide real-time status of all doors.", "label": 0},

    # Non-Functional/Ambiguous (Label 1)
    {"requirement_text": "The system must respond to user commands within 2 seconds.", "label": 1},
    {"requirement_text": "All user data shall be encrypted using AES-256 encryption.", "label": 1},
    {"requirement_text": "The system shall maintain 99.9% uptime.", "label": 1},
    {"requirement_text": "The mobile application shall support both iOS and Android.", "label": 1},
    {"requirement_text": "The system shall operate under temperatures from -10 to 50 degrees Celsius.", "label": 1},
    {"requirement_text": "The system shall log events securely without data loss.", "label": 1},
    {"requirement_text": "The application shall provide automatic updates without downtime.", "label": 1},
    {"requirement_text": "The system shall handle up to 500 simultaneous users.", "label": 1},
    {"requirement_text": "The system shall be compatible with multiple smart home devices.", "label": 1},
    {"requirement_text": "The system shall maintain all data for at least one year for auditing.", "label": 1},
    {"requirement_text": "The system must be easy to use.", "label": 1},
    {"requirement_text": "The security must be good.", "label": 1},
    {"requirement_text": "The application should be fast.", "label": 1},
    {"requirement_text": "The system should perform well.", "label": 1},
    {"requirement_text": "Users must find the interface user-friendly.", "label": 1},
]

# تطبيق التنظيف على البيانات الأصلية
for d in seed_data:
    d["requirement_text"] = clean_text(d["requirement_text"])

# ------------------------------
# 2) Generation Utilities
# ------------------------------
doors = ["front door", "garage door", "back door", "main entrance", "side entrance", "bedroom door", "patio door"]
auth_methods = ["biometric fingerprint", "face recognition", "PIN code", "password + 2FA", "NFC tag", "RFID card"]
guest_flows = ["temporary guest code", "time-limited access token", "one-time access link", "scheduled entry permit"]
events = ["entry", "exit", "failed authentication", "tamper alert", "forced entry", "power failure", "network loss"]
metrics = ["response time", "throughput", "success rate", "time-to-authenticate", "latency", "failure rate"]
numbers = ["1", "2", "3", "5", "10", "30"]
concurrency_values = ["50", "100", "200", "500", "1000", "2500"]
vague_adjectives = ["quick", "easy", "simple", "good", "nice", "user-friendly", "robust", "intuitive", "reliable", "secure enough", "fast enough"]
vague_verbs = ["perform well", "handle gracefully", "be reliable", "be secure", "work efficiently", "be stable"]
security_metrics = ["AES-256", "SHA-512", "TLS 1.3", "multi-factor authentication"]
time_metrics = ["milliseconds", "seconds", "minutes"]
openers = ["It is required that the system", "The product shall", "The application must", "The smart lock system will", "Users should be able to"]

def choose(seq): return random.choice(seq)
def make_functional():
    return f"{choose(openers)} allow an authorized user to unlock the {choose(doors)} via {choose(auth_methods)}."

def make_nonfunc():
    return f"{choose(openers)} maintain {choose(metrics)} under {choose(numbers)} {choose(time_metrics)} under typical load."

def make_ambiguous():
    return f"The interface must be very {choose(vague_adjectives)} and should {choose(vague_verbs)}."

# ------------------------------
# 3) Generate Dataset + Cleaning
# ------------------------------
def generate_smart_home_requirements(target_total=1000, dist=(0.5, 0.5)):
    func_ratio, nonfunc_ratio = dist
    func_count = int(target_total * func_ratio)
    nonfunc_count = target_total - func_count

    generated = []
    for _ in range(func_count):
        generated.append({"requirement_text": clean_text(make_functional()), "label": 0})
    for _ in range(nonfunc_count):
        generated.append({"requirement_text": clean_text(make_nonfunc()), "label": 1})

    return generated

generated = generate_smart_home_requirements(target_total=1000)
all_data = seed_data + generated
random.shuffle(all_data)

# ------------------------------
# 4) Dataset Preparation
# ------------------------------
dataset = Dataset.from_list(all_data)
split = dataset.train_test_split(test_size=0.12, seed=RANDOM_SEED)
train_dataset = split["train"]
eval_dataset = split["test"]

# ------------------------------
# 5) Tokenizer & Model (RoBERTa)
# ------------------------------
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def tokenize(batch):
    return tokenizer(batch["requirement_text"], padding=True, truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
eval_dataset = eval_dataset.map(tokenize, batched=True)
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# ------------------------------
# 6) Metrics
# ------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")
    return {"accuracy": acc, "f1_binary": f1}

# ------------------------------
# 7) Training
# ------------------------------

training_args = TrainingArguments(
    output_dir="./finetuned_model",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_binary",
    greater_is_better=True,
    logging_steps=50,
    save_total_limit=2,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("🚀 Starting training (clean dataset)...")
trainer.train()
print("✅ Training finished successfully.")

# ------------------------------
# 8) Save Model
# ------------------------------
trainer.save_model("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")

# ------------------------------
# 9) Test Example Predictions
# ------------------------------
classifier = pipeline("text-classification", model="./finetuned_model", tokenizer="./finetuned_model", device=-1)
test_texts = [
    "The system shall unlock the front door via biometric fingerprint.",
    "The system must maintain response time under 2 seconds.",
    "The interface should be very simple for all users."
]

print("\n🔍 Test Predictions:")
for t in test_texts:
    res = classifier(t)
    label_id = int(res[0]["label"].replace("LABEL_", ""))
    label_map = {0: "Functional", 1: "Non-Functional/Ambiguous"}
    print(f"- \"{t}\" → {label_map[label_id]} (score={res[0]['score']:.2f})")