import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import evaluate
import numpy as np

# -------------------------------
# 1. Load Dataset
# -------------------------------
dataset = load_dataset("csv", data_files="data/data.csv")["train"]
dataset = dataset.train_test_split(test_size=0.1, seed=42)

label_map = {"negative": 0, "neutral": 1, "positive": 2}

def encode_labels(example):
    example["label"] = label_map[example["Sentiment"].lower()]
    return example

dataset = dataset.map(encode_labels)

# -------------------------------
# 2. Tokenize
# -------------------------------
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(batch):
    return tokenizer(batch["Sentence"], padding="max_length", truncation=True, max_length=128)

encoded = dataset.map(preprocess, batched=True)
encoded = encoded.rename_column("label", "labels")
encoded.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# -------------------------------
# 3. Load Base Model + LoRA
# -------------------------------
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["query", "key", "value"]
)

model = get_peft_model(model, lora_config)

# -------------------------------
# 4. Metrics
# -------------------------------
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"]
    }

# -------------------------------
# 5. Training
# -------------------------------
training_args = TrainingArguments(
    output_dir="models/fin_lora_adapter",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=20,
    report_to="none",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

model.save_pretrained("models/fin_lora_adapter")
tokenizer.save_pretrained("models/fin_lora_adapter")
