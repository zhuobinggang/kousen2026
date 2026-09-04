import math
import MeCab
from collections import Counter
import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. 日语分词器准备 (提取原型/基本形)
tagger = MeCab.Tagger()

def tokenize_ja(text):
    node = tagger.parseToNode(text)
    tokens = []
    while node:
        # 过滤空白和未命中的节点
        if node.surface.strip():
            features = node.feature.split(",")
            pos = features[0]  # 品词
            
            # 过滤掉纯标点符号（补助记号）
            if pos != "補助記号":
                # features[10] 为 UniDic 的汉字规范原型（代表表記）
                if len(features) > 10 and features[10] != "*" and features[10].strip():
                    lemma = features[10]
                else:
                    lemma = node.surface  # 兜底：取原始表层形
                
                # 去除 UniDic 可能附带的词性后缀（如 "-する"）并防空串
                lemma = lemma.split("-")[0].strip()
                if lemma:
                    tokens.append(lemma)
                    
        node = node.next
    return tokens


# 2. 手动实现 TF-IDF 特征提取器
class ManualTfidfVectorizer:
    def __init__(self):
        self.vocab = {}          # 单词 -> 特征列索引
        self.idf_diag = None     # 每个词对应的 IDF 权重
        
    def fit_transform(self, raw_documents):
        # 步骤 A: 分词
        tokenized_docs = [tokenize_ja(doc) for doc in raw_documents]
        n_samples = len(tokenized_docs)
        
        # 步骤 B: 构建词表 (Vocabulary)
        unique_words = sorted(list({word for doc in tokenized_docs for word in doc}))
        self.vocab = {word: idx for idx, word in enumerate(unique_words)}
        vocab_size = len(self.vocab)
        
        # 步骤 C: 统计文档频率 (DF) 并计算 IDF
        # 采用与 scikit-learn 一致的平滑公式: log((1 + n) / (1 + df)) + 1
        df_counts = Counter()
        for doc in tokenized_docs:
            df_counts.update(set(doc))
            
        self.idf_diag = np.zeros(vocab_size)
        for word, idx in self.vocab.items():
            df = df_counts[word]
            self.idf_diag[idx] = math.log((1.0 + n_samples) / (1.0 + df)) + 1.0
            
        # 步骤 D: 计算各样本的 TF-IDF 矩阵
        return self._transform_tokens(tokenized_docs)
    
    def transform(self, raw_documents):
        """用于测试集或新文本的向量转换"""
        tokenized_docs = [tokenize_ja(doc) for doc in raw_documents]
        return self._transform_tokens(tokenized_docs)

    def _transform_tokens(self, tokenized_docs):
        n_samples = len(tokenized_docs)
        vocab_size = len(self.vocab)
        X = np.zeros((n_samples, vocab_size), dtype=np.float32)
        
        for row_idx, doc in enumerate(tokenized_docs):
            if not doc:
                continue
            word_counts = Counter(doc)
            
            # 填入原始词频 (Term Frequency) 并乘以已计算好的 IDF
            for word, count in word_counts.items():
                if word in self.vocab:
                    col_idx = self.vocab[word]
                    X[row_idx, col_idx] = count * self.idf_diag[col_idx]
                    
            # 步骤 E: L2 归一化 (让向量模长变为 1)
            norm = np.linalg.norm(X[row_idx])
            if norm > 0:
                X[row_idx] /= norm
                
        return X


def build_vocab_and_fit(train_texts, train_labels):
    # 提取特征
    vectorizer = ManualTfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    y_train = np.array(train_labels)

    print(f"词表大小: {len(vectorizer.vocab)}")
    print(f"X_train 形状: {X_train.shape}")  # (4, 词表大小)

    # 训练逻辑回归分类器
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # 查看实际停止时跑的迭代次数
    print("实际迭代轮数:", clf.n_iter_[0])
    return vectorizer, clf

def predict_sentiment(vectorizer, clf, new_texts):
    X_new = vectorizer.transform(new_texts)
    predictions = clf.predict(X_new)
    return predictions

# 3. 运行测试与接入 LogisticRegression
if __name__ == "__main__":
    # 模拟训练数据 (1: 积极, 0: 消极)
    train_texts = [
        "この映画は本当に面白くて感動しました。",
        "最高のサービスで、とても満足しています！",
        "全然面白くなかった、時間が無駄だった。",
        "対応が悪くて非常にがっかりしました。"
    ]
    train_labels = [1, 1, 0, 0]

    vectorizer, clf = build_vocab_and_fit(train_texts, train_labels)

    # 测试新样本
    test_texts = [
        "とても面白くて最高でした！",
        "サービスが悪くてがっかりした。"
    ]
    predictions = predict_sentiment(vectorizer, clf, test_texts)

    for text, pred in zip(test_texts, predictions):
        label_str = "正面 (Positive)" if pred == 1 else "负面 (Negative)"
        print(f"文本: \"{text}\" -> 预测结果: {label_str}")