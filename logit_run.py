from logit import *
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

# 1. 定义需要保留的列名
target_columns = [
    'Sentence',
    'Writer_Sadness'
]

# 2. 读取 TSV 文件
df = pd.read_csv(
    './wrime/wrime-ver2.tsv',
    sep='\t',
    usecols=target_columns
)

# 过滤可能存在的空行
df = df.dropna(subset=['Sentence', 'Writer_Sadness'])

print(df.head())
print(f"总样本数: {len(df)}")
sadness_count = (df['Writer_Sadness'] > 0).sum()
print(f"Writer_Sadness > 0 的样本数: {sadness_count} (占比: {sadness_count / len(df):.2%})")

# 3. 提取文本与标签 (WRIME 情感强度 0~3，二值化为 0:无悲伤, 1:有悲伤)
texts = df['Sentence'].tolist()
labels = [1 if val > 0 else 0 for val in df['Writer_Sadness']]

# 4. 按照 8:1:1 比例划分训练集、验证集和测试集
n_total = len(texts)
train_end = int(0.8 * n_total)
val_end = int(0.9 * n_total)

train_texts, train_labels = texts[:train_end], labels[:train_end]
val_texts, val_labels = texts[train_end:val_end], labels[train_end:val_end]
test_texts, test_labels = texts[val_end:], labels[val_end:]

print(f"划分结果 -> 训练集: {len(train_texts)}, 验证集: {len(val_texts)}, 测试集: {len(test_texts)}")

# 5. 特征提取与模型训练
vectorizer, clf = build_vocab_and_fit(train_texts, train_labels)

# 6. 在验证集上评估
val_preds = predict_sentiment(vectorizer, clf, val_texts)
print("\n=== 验证集评估结果 ===")
print(f"Validation Accuracy: {accuracy_score(val_labels, val_preds):.4f}")
print(f"Validation F1-Score: {f1_score(val_labels, val_preds, average='macro'):.4f}")

# 7. 在测试集上做最终评估
test_preds = predict_sentiment(vectorizer, clf, test_texts)
print("\n=== 测试集最终评估 ===")
print(f"Test Accuracy: {accuracy_score(test_labels, test_preds):.4f}")
print("\n详细分类报告 (Test Set):")
print(classification_report(test_labels, test_preds, target_names=["非悲伤 (0)", "悲伤 (1)"]))

# 8. 实时推理示例
sample_sentences = [
    "試験に落ちてしまって、本当に泣きたい。",
    "今日は友達と遊園地に行ってすごく楽しかった！"
]
sample_preds = predict_sentiment(vectorizer, clf, sample_sentences)

print("\n=== 实时预测测试 ===")
for text, pred in zip(sample_sentences, sample_preds):
    label_desc = "悲伤 (Sadness)" if pred == 1 else "非悲伤 (Not Sadness)"
    print(f"输入: \"{text}\" -> 预测: {label_desc}")