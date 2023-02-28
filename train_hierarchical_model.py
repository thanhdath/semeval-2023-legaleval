import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from transformers import AutoTokenizer
import torch
import json
from transformers import (
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import (
    classification_report,
)
from multiprocessing import Pool
import argparse
import os
import pickle as pkl
import evaluate

accuracy = evaluate.load("accuracy")

# import bitsandbytes as bnb
from torch import nn
from transformers.trainer_pt_utils import get_parameter_names

# from hierarchical_longformer import HierarchicalLongformerForSequenceClassification
from model.hierarchical_longformer_wsum import (
    HierarchicalAttentionLongformerForSequenceClassification,
)

# from hierarchical_bert import HierarchicalBertForSequenceClassification
from data_collator.data_collator_chunking import DataCollatorChunking
import re

parser = argparse.ArgumentParser()
parser.add_argument("--max-length", default=512, type=int)
parser.add_argument(
    "--model-path", default="/home/datht17/huggingface/legalbert-base-uncased/"
)
parser.add_argument("--save-dir", default="models/temp/legalbert-base-uncased")
parser.add_argument("--log-dir", default="results/temp/legalbert-base-uncased")
parser.add_argument("--truncation-side", default="left", choices=["left", "right"])
parser.add_argument("--epochs", default=5, type=int)
parser.add_argument("--batch-size", default=8, type=int, help="Document batch size")
parser.add_argument("--lr", default=2e-5, type=float, help="Learning rate")
parser.add_argument("--gradient-accumulation-steps", default=8, type=int)
parser.add_argument("--chunk-overlap", default=100, type=int)
parser.add_argument("--max-n-chunks", default=2, type=int)
parser.add_argument("--n-processes", default=32, type=int)
parser.add_argument("--dataset", default="semeval", choices=["ildc", "semeval"])

args = parser.parse_args()

print(args)


tokenizer = AutoTokenizer.from_pretrained(args.model_path)


if args.dataset == "ildc":
    print("Load dataset ILDC")
    ildc_single = pd.read_csv("data/ILDC/ILDC_single/ILDC_single.csv")
    ildc_multi = pd.read_csv("data/ILDC/ILDC_multi/ILDC_multi.csv")
elif args.dataset == "semeval":
    print("Load dataset SemEval")
    ildc_single = pd.read_csv("data/subtask3/ILDC_single_train_dev.csv")
    ildc_multi = pd.read_csv("data/subtask3/ILDC_multi_train_dev.csv")

print("Split:", set(ildc_single["split"].values))

ori_train_data = []
ori_dev_data = []
ori_test_data = []

for text, label, split in np.concatenate(
    [ildc_single.values, ildc_multi.values], axis=0
)[:, :3]:
    text = text.replace("\n", " ")
    text = re.sub("\s+", " ", text).strip()

    # if 'uncased' in args.model_path or args.dataset == 'semeval':
    text = text.lower()

    if split == "train":
        ori_train_data.append({"text": text, "label": label})
    elif split == "dev":
        ori_dev_data.append({"text": text, "label": label})
    elif split == "test":
        ori_test_data.append({"text": text, "label": label})

if len(ori_test_data) == 0:
    ori_test_data = ori_dev_data


def split_into_chunks(params):
    example, doc_id, chunk_size, overlap = params

    input_ids = tokenizer(example["text"])["input_ids"]

    chunks = []
    attention_masks = []
    doc_ids = []

    i = 0
    input_ids = input_ids[1:-1]  # remove cls and sep tokens
    input_ids = input_ids[
        ::-1
    ]  # reverse order, -> max sure last chunks always have 4096 tokens
    while i < len(input_ids):
        chunk = input_ids[i : i + chunk_size - 2]  # -2 special tokens cls and sep
        chunk = chunk[::-1]
        next_i = i + chunk_size - 2

        chunk = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
        i = next_i - overlap

        attention_mask = [1] * len(chunk)
        # padding to max_length
        if len(chunk) < chunk_size:
            pad_size = chunk_size - len(chunk)
            chunk += [tokenizer.pad_token_id] * pad_size
            attention_mask += [0] * pad_size

        assert (
            len(attention_mask) == args.max_length
        ), f"attention_mask {len(attention_mask)} - chunk_size {chunk_size} - len chunk {len(chunk)}"

        chunks.append(chunk)
        attention_masks.append(attention_mask)
        doc_ids.append(doc_id)

    chunks = chunks[::-1]
    attention_masks = attention_masks[::-1]
    doc_ids = doc_ids[::-1]

    MAX_N_CHUNKS = args.max_n_chunks
    if len(chunks) > MAX_N_CHUNKS:
        chunks = chunks[-MAX_N_CHUNKS:]
        attention_masks = attention_masks[-MAX_N_CHUNKS:]
        doc_ids = doc_ids[-MAX_N_CHUNKS:]

    features = {
        "chunks": chunks,
        "attention_masks": attention_masks,
        "doc_ids": doc_ids,
    }

    if "label" in example:
        features["label"] = example["label"]

    return features


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


pool = Pool(args.n_processes)
tokenized_train_data = pool.map(
    split_into_chunks,
    [(x, i, args.max_length, args.chunk_overlap) for i, x in enumerate(ori_train_data)],
)
tokenized_dev_data = pool.map(
    split_into_chunks,
    [(x, i, args.max_length, args.chunk_overlap) for i, x in enumerate(ori_dev_data)],
)
tokenized_test_data = pool.map(
    split_into_chunks,
    [(x, i, args.max_length, args.chunk_overlap) for i, x in enumerate(ori_test_data)],
)
pool.close()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = HierarchicalAttentionLongformerForSequenceClassification.from_pretrained(
    args.model_path
).to(device)

data_collator = DataCollatorChunking()

training_args = TrainingArguments(
    output_dir=args.log_dir,
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=0.01,
    gradient_checkpointing=True,
    fp16=True,
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
