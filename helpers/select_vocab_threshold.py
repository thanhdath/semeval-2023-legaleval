
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import pandas as pd

# ildc_single = pd.read_csv('data/subtask3/ILDC_single_train_dev.csv')
# ildc_multi = pd.read_csv('data/subtask3/ILDC_multi_train_dev.csv')
ildc_single = pd.read_csv('data/ILDC/ILDC_single/ILDC_single.csv')
ildc_multi = pd.read_csv('data/ILDC/ILDC_multi/ILDC_multi.csv')


df = pd.concat([ildc_single, ildc_multi])

df['text'] = df['text'].apply(lambda x: x.lower())

train_df = df[df['split'] == 'train']
train_df


# In[18]:


super_text = " ".join(train_df['text'].values) #.lower()

words = super_text.split()
from collections import Counter
word_freqs = Counter(words)


# In[19]:


sorted_freq = sorted(word_freqs.items(), key=lambda x: x[1], reverse=True)


# In[20]:


len(sorted_freq)


# In[27]:


import pickle as pkl
from tqdm import tqdm

params = []

vectorizers = []

for threshold_freq in tqdm([100, 150, 200, 250, 300, 350, 400, 450, 500]):
    # truncate K most popular words and remove words that occurs less than threshold_freq

    vocabs = [x for x, f in sorted_freq[1000:] if f >= threshold_freq]
    # train a vectorizer
    
    print(f'Threshold freq: {threshold_freq} - vocab size: {len(vocabs)}')

    vectorizer = TfidfVectorizer(vocabulary=vocabs)
    vectorizer.fit(train_df['text'].values)
    vectorizers.append(vectorizer)

    train_vector = vectorizer.transform(train_df['text'].values)
    train_label = train_df['label'].values
    dev_vector = vectorizer.transform(df[df['split'] == 'dev']['text'].values)
    dev_label = df[df['split'] == 'dev']['label'].values
    # test_vector = vectorizer.transform(df[df['split'] == 'test']['text'].values)

    from sklearn.svm import LinearSVC

    model = LinearSVC()
    model.fit(train_vector, train_label)

    predict = model.predict(dev_vector)

    from sklearn.metrics import (
        classification_report
    )

    print(classification_report(dev_label, predict))
    
    pkl.dump(vectorizer, open(f'tfidf_vectorizer-threshold{threshold_freq}.pkl', 'wb'))

