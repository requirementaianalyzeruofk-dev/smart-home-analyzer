from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
from datasets import Dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# بيانات التدريب (Smart Lock Requirements)
data = [
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
]

# تحويل البيانات وتقسيمها
dataset = Dataset.from_list(data)
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# تحميل النموذج الأساسي والمحول
model_name = "bert-base-uncased" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# دالة التحويل إلى رموز
def tokenize(batch):
    return tokenizer(batch["requirement_text"], padding=True, truncation=True)

train_dataset = train_dataset.map(tokenize, batched=True)
eval_dataset = eval_dataset.map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# دالة حساب المقاييس
def compute_metrics(p):
    logits, labels = p
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": acc, "f1_score": f1}

# إعدادات التدريب
training_args = TrainingArguments(
    output_dir="./finetuned_model",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    load_best_model_at_end=True,
)

# تنفيذ التدريب
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# حفظ النموذج والمحول النهائي في المجلد finetuned_model
trainer.save_model("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")


# تجربة التصنيف بعد التدريب
classifier = pipeline(
    "text-classification",
    model="./finetuned_model", 
    tokenizer="./finetuned_model",
    device=-1
)

test_text = "The system shall allow users to unlock doors remotely using the app."
result = classifier(test_text)
label_id = int(result[0]['label'].replace('LABEL_', ''))
label_map = {0: "Functional", 1: "Non-Functional"}
label = label_map.get(label_id, "Unknown")
score = result[0]['score']

print("\n==================================")
print("اختبار النموذج المُدَرَّب")
print("==================================")
print(f"النص: {test_text}")
print(f"التصنيف المُتوقَّع: {label}, درجة الثقة: {score:.2%}")