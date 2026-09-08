# 只使用bagofword不使用tfidf

import MeCab
import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. 日语分词器准备 (提取原型/基本形，并保留核心实词)
tagger = MeCab.Tagger()

# 0/1 词袋下缺少 IDF 降权，建议重点保留有情感倾向的实词品词
TARGET_POS = {"名詞", "動詞", "形容詞", "形状詞", "副詞"}

def tokenize_ja(text):
    node = tagger.parseToNode(text)
    tokens = []
    while node:
        if node.surface.strip():
            features = node.feature.split(",")
            pos = features[0]  # 品词
            
            # 只保留核心实词，过滤掉标点、助词、助动词等噪声
            if pos in TARGET_POS:
                # features[10] 为 UniDic 的规范原型（代表表記）
                if len(features) > 10 and features[10] != "*" and features[10].strip():
                    lemma = features[10]
                else:
                    lemma = node.surface  # 兜底：取原始表层形
                
                # 去除 UniDic 可能附带的词性后缀（如 "-する"）
                lemma = lemma.split("-")[0].strip()
                if lemma:
                    tokens.append(lemma)
                    
        node = node.next
    return tokens


# 2. 手动实现 0/1 (Binary) Bag-of-Words 特征提取器
class ManualBinaryBagOfWords:
    def __init__(self):
        self.vocab = {}  # 单词 -> 特征列索引
        
    def fit_transform(self, raw_documents):
        # 步骤 A: 分词
        tokenized_docs = [tokenize_ja(doc) for doc in raw_documents]
        
        # 步骤 B: 构建词表 (Vocabulary)
        unique_words = sorted(list({word for doc in tokenized_docs for word in doc}))
        self.vocab = {word: idx for idx, word in enumerate(unique_words)}
        
        # 步骤 C: 生成 0/1 矩阵
        return self._transform_tokens(tokenized_docs)
    
    def transform(self, raw_documents):
        """用于测试集或新文本的向量转换"""
        tokenized_docs = [tokenize_ja(doc) for doc in raw_documents]
        return self._transform_tokens(tokenized_docs)

    def _transform_tokens(self, tokenized_docs):
        n_samples = len(tokenized_docs)
        vocab_size = len(self.vocab)
        # 初始化全 0 矩阵
        X = np.zeros((n_samples, vocab_size), dtype=np.float32)
        
        for row_idx, doc in enumerate(tokenized_docs):
            # 使用 set(doc) 去重：出现即为 1，不计重复词频
            for word in set(doc):
                if word in self.vocab:
                    col_idx = self.vocab[word]
                    X[row_idx, col_idx] = 1.0
                    
        return X


# 3. 训练与预测包装函数
def build_vocab_and_fit(train_texts, train_labels):
    # 提取 0/1 词袋特征
    vectorizer = ManualBinaryBagOfWords()
    X_train = vectorizer.fit_transform(train_texts)
    y_train = np.array(train_labels)

    print(f"词表大小: {len(vectorizer.vocab)}")
    print(f"词表内容: {list(vectorizer.vocab.keys())}")
    print(f"X_train 形状: {X_train.shape}")

    # 训练逻辑回归分类器
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    print("实际迭代轮数:", clf.n_iter_[0])
    return vectorizer, clf

def predict_sentiment(vectorizer, clf, new_texts):
    X_new = vectorizer.transform(new_texts)
    predictions = clf.predict(X_new)
    return predictions


# 4. 测试运行
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