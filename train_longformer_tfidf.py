import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from transformers import AutoTokenizer
import torch
from datasets import Dataset
import json
from transformers import TrainingArguments, Trainer
from sklearn.metrics import (
    classification_report,
)
import argparse
import re
import pickle as pkl
from model.longformer_tfidf import LongformerTFIDFForSequenceClassification
from data_collator.data_collator_tfidf import DataCollatorTFIDF
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
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--seed", default=100, type=int)
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--dataset", default="semeval", choices=["semeval", "ildc"])
parser.add_argument(
    "--tfidf-vectorizer", type=str, default="tfidf_vectorizer-threshold350.pkl"
)
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

train_data = []
dev_data = []
test_data = []

if args.dataset == "ildc":
    ildc_single = pd.read_csv("data/ILDC/ILDC_single/ILDC_single.csv")
    ildc_multi = pd.read_csv("data/ILDC/ILDC_multi/ILDC_multi.csv")
else:
    ildc_single = pd.read_csv("data/subtask3/ILDC_single_train_dev.csv")
    ildc_multi = pd.read_csv("data/subtask3/ILDC_multi_train_dev.csv")

print("Split:", set(ildc_single["split"].values))

for text, label, split in np.concatenate(
    [ildc_single.values, ildc_multi.values], axis=0
)[:, :3]:
    text = text.replace("\n", " ")
    text = re.sub("\s+", " ", text).strip()
    text = text.lower()

    if split == "train":
        train_data.append({"text": text, "label": label})
    elif split == "dev":
        dev_data.append({"text": text, "label": label})
    elif split == "test":
        test_data.append({"text": text, "label": label})

if len(test_data) == 0:
    test_data = dev_data

# train_data = json.load(open(train_path))
# test_data = json.load(open(test_path))

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
    predictions = torch.argmax(logits[0], dim=1)
    return predictions, labels


def preprocess_function(examples):
    text = examples["text"]
    return_dict = tokenizer(text, truncation=True, max_length=args.max_length)

    tfidf_vector = np.array(vectorizer.transform(text).todense())
    return_dict["tfidf_feature"] = tfidf_vector
    return return_dict


vectorizer = pkl.load(open(args.tfidf_vectorizer, "rb"))

tokenized_train_data = train_data.map(preprocess_function, batched=True, num_proc=32)
tokenized_dev_data = dev_data.map(preprocess_function, batched=True, num_proc=32)
tokenized_test_data = test_data.map(preprocess_function, batched=True, num_proc=32)

data_collator = DataCollatorTFIDF(tokenizer=tokenizer)

model = LongformerTFIDFForSequenceClassification.from_pretrained(
    args.model, num_labels=2, ignore_mismatched_sizes=True
).to(device)

training_args = TrainingArguments(
    output_dir=args.log_dir,
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=0.01,
    gradient_checkpointing=True,
    fp16=args.fp16,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
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
