import pandas as pd

target_columns = [
    'Sentence',
    'Writer_Joy'
]

df = pd.read_csv(
    './wrime/wrime-ver2.tsv',
    sep='\t',
    nrows=20000, 
    usecols=target_columns
)

sentences = []
labels = []
for index, row in df.iterrows():
    sentences.append(row['Sentence'])
    labels.append(1 if row['Writer_Joy'] > 0 else 0)

train_sentences = sentences[:18000]
test_sentences = sentences[18000:]
train_labels = labels[:18000]
test_labels = labels[18000:]

# DONE
vocab = texts2vocab(train_sentences) # 分詞、sort、辞書を作る
train_sentence_bows = texts2bows(vocab, train_sentences) # 18000個のbowリストを得る
test_sentence_bows = texts2bows(vocab, test_sentences)


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

# TODO: train_sentence_bows, test_sentence_bows, train_labels, test_labelsを
# numpy arrayにする

# 1. 訓練
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(train_sentence_bows, train_labels)

# 2. 予測
test_preds = clf.predict(test_sentence_bows)

f1_macro = f1_score(test_labels, test_preds, average='macro')
print(f"F1-score (Macro): {f1_macro:.4f}")

print("\n詳細報告:")
print(classification_report(test_labels, test_preds, target_names=["Not Joy (0)", "Joy (1)"]))