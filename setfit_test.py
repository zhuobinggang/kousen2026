import pandas as pd
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. 读取数据
target_columns = ['Sentence', 'Writer_Joy']
df = pd.read_csv(
    './wrime/wrime-ver2.tsv',
    sep='\t',
    nrows=200, 
    usecols=target_columns
)

# 2. 构造文本与标签（二分类：0 或 1）
df['label'] = (df['Writer_Joy'] > 0).astype(int)
df = df.rename(columns={'Sentence': 'text'})

# 划分训练集与测试集
train_df = df.iloc[:180][['text', 'label']]
test_df = df.iloc[180:][['text', 'label']]

train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

# 3. 初始化 SetFit 模型
# 推荐使用名古屋大学开源的高质量日语表征模型，或多语言 paraphrase-multilingual-mpnet-base-v2
model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SetFitModel.from_pretrained(model_id)

# 4. 配置训练参数
args = TrainingArguments(
    batch_size=16,
    num_epochs=1,              # 对比学习迭代轮数，小数据集通常 1-2 轮即可
    num_iterations=20,         # 每个样本采样的正负对数量（生成 180 * 20 对样本）
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=50,
    load_best_model_at_end=True,
    learning_rate=2e-5,
)

def compute_metrics(y_pred, y_test):
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    acc = accuracy_score(y_test, y_pred)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# 5. 训练与评估
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    metric=compute_metrics,
)

trainer.train()
metrics = trainer.evaluate()
print("测试集评估结果:", metrics)

# 6. 推理预测示例
test_texts = [
    "今日はずっと行きたかったカフェに行けて本当に嬉しかった！",
    "締め切りに追われて胃が痛い一日だった。"
]
predictions = model.predict(test_texts)
print("预测结果 (1: Joy, 0: Non-Joy):", predictions)

# 7. 保存模型（可选）
# model.save_pretrained("./setfit-wrime-joy-model")