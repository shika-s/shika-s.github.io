import warnings

import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from evaluate import load
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

load_dotenv()
warnings.filterwarnings("ignore")

task = "sst2"
model_checkpoint = "distilbert-base-uncased"
batch_size = 512

dataset = load_dataset("glue", task)
metric = load("glue", task)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)


def preprocess(examples):
    return tokenizer(examples["sentence"], truncation=True)


encoded_dataset = dataset.map(preprocess, batched=True)

label2id = {"NEGATIVE": 0, "POSITIVE": 1}
id2label = {v: k for k, v in label2id.items()}
config = AutoConfig.from_pretrained(
    model_checkpoint, label2id=label2id, id2label=id2label
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, config=config
)

model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-finetuned-{task}",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.06,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    bf16=True,
    tf32=True,
    torch_compile=True,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    seed=42,
    data_seed=42,
    push_to_hub=True,
    save_safetensors=True,
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()
trainer.evaluate()
trainer.push_to_hub()

# cleanup: remove files not needed for inference
from huggingface_hub import HfApi

api = HfApi()
repo_id = f"winegarj/{model_name}-finetuned-{task}"
for f in api.list_repo_files(repo_id):
    if f in ("training_args.bin", "pytorch_model.bin"):
        api.delete_file(f, repo_id=repo_id)
