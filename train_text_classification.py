import numpy as np
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from transformers import AutoTokenizer
import torch
from datasets import Dataset
import json
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import (
    classification_report,
)
import argparse
import re
import evaluate

accuracy = evaluate.load("accuracy")

parser = argparse.ArgumentParser()
parser.add_argument("--max-length", default=4096, type=int)
parser.add_argument(
    "--model", default="/home/datht17/huggingface/longformer-large-4096"
)
parser.add_argument("--save-dir", default="models/longformer-large-finetuned")
parser.add_argument("--log-dir", default="results/longformer-large-finetuned")
parser.add_argument("--truncation-side", default="left", choices=["left", "right"])
parser.add_argument("--batch-size", default=8, type=int)
parser.add_argument("--gradient-accumulation-steps", default=8, type=int)
parser.add_argument("--epochs", default=5, type=int)
parser.add_argument("--dataset", default="semeval", choices=["ildc", "semeval"])
args = parser.parse_args()

print(args)

train_data = []
dev_data = []
test_data = []

if args.dataset == "ildc":
    print("Load dataset ILDC")
    ildc_single = pd.read_csv("data/ILDC/ILDC_single/ILDC_single.csv")
    ildc_multi = pd.read_csv("data/ILDC/ILDC_multi/ILDC_multi.csv")
elif args.dataset == "semeval":
    print("Load dataset SemEval")
    ildc_single = pd.read_csv("data/subtask3/ILDC_single_train_dev.csv")
    ildc_multi = pd.read_csv("data/subtask3/ILDC_multi_train_dev.csv")

for text, label, split in np.concatenate(
    [ildc_single.values, ildc_multi.values], axis=0
)[:, :3]:
    text = text.replace("\n", " ")
    text = re.sub("\s+", " ", text).strip()

    #     if 'uncased' in args.model or args.dataset == 'semeval':
    text = text.lower()

    if split == "train":
        train_data.append({"text": text, "label": label})
    elif split == "dev":
        dev_data.append({"text": text, "label": label})
    elif split == "test":
        test_data.append({"text": text, "label": label})

if len(test_data) == 0:
    test_data = dev_data

train_data = Dataset.from_list(train_data)
dev_data = Dataset.from_list(dev_data)
test_data = Dataset.from_list(test_data)

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(
    args.model, truncation_side=args.truncation_side
)


def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    predictions = eval_pred.predictions[0]
    return accuracy.compute(predictions=predictions, references=labels)


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    predictions = torch.argmax(logits, dim=1)
    return predictions, labels


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=args.max_length)


tokenized_train_data = train_data.map(preprocess_function, batched=True, num_proc=32)
tokenized_dev_data = dev_data.map(preprocess_function, batched=True, num_proc=32)
tokenized_test_data = test_data.map(preprocess_function, batched=True, num_proc=32)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(
    device
)

training_args = TrainingArguments(
    output_dir=args.log_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=0.01,
    gradient_checkpointing=True,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_dev_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

print("start training")
trainer.train()
trainer.save_model(args.save_dir)

predict_result = trainer.predict(tokenized_test_data)
predict_result = predict_result.predictions[0]

y_test = [x["label"] for x in tokenized_test_data]
print(classification_report(y_test, predict_result))
