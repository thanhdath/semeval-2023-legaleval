from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import argparse
import pickle as pkl
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="semeval", choices=["ildc", "semeval"])
parser.add_argument("--frequency-threshold", default=100, type=int)
args = parser.parse_args()


if args.dataset == "semeval":
    ildc_single = pd.read_csv("data/subtask3/ILDC_single_train_dev.csv")
    ildc_multi = pd.read_csv("data/subtask3/ILDC_multi_train_dev.csv")
elif args.dataset == "ildc":
    ildc_single = pd.read_csv("data/ILDC/ILDC_single/ILDC_single.csv")
    ildc_multi = pd.read_csv("data/ILDC/ILDC_multi/ILDC_multi.csv")
else:
    raise NotImplementedError()

df = pd.concat([ildc_single, ildc_multi])
df["text"] = df["text"].apply(lambda x: x.lower())

# Sort word frequencies
train_df = df[df["split"] == "train"]
super_text = " ".join(train_df["text"].values)  #

words = super_text.split()
word_freqs = Counter(words)
sorted_freq = sorted(word_freqs.items(), key=lambda x: x[1], reverse=True)

# Truncate rare words and stop-words
vocabs = [x for x, f in sorted_freq[1000:] if f >= args.frequency_threshold]
print(f"Threshold freq: {args.frequency_threshold} - vocab size (tfidf_feature_size): {len(vocabs)}")

# train a vectorizer
vectorizer = TfidfVectorizer(vocabulary=vocabs)
vectorizer.fit(train_df["text"].values)

pkl.dump(
    vectorizer,
    open(
        f"{args.dataset}-tfidf_vectorizer-threshold{args.frequency_threshold}.pkl", "wb"
    ),
)
