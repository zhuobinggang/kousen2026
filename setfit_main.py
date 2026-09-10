import pandas as pd
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ================= 自分のデータ =================
# sentences: ["今日の天気がいい。", "ご飯を食べましょう。"]
train_sentences = ["今日の天気がいい。", "ご飯を食べましょう。"]  # 0~99
test_sentences = ["今日の天気がいい。", "ご飯を食べましょう。"]   # 100~199

# labels: [1, 0]
train_labels = [1,0]     # 0~99
test_labels = [1,0]      # 100~199
# ===============================================

train_dataset = Dataset.from_dict({
    "text": train_sentences,
    "label": train_labels
})

test_dataset = Dataset.from_dict({
    "text": test_sentences,
    "label": test_labels
})

model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SetFitModel.from_pretrained(model_id)

args = TrainingArguments(
    batch_size=8,
    num_epochs=1,
    num_iterations=10,
    body_learning_rate=2e-5,   
    eval_strategy="steps",    
    eval_steps=50,
    save_steps=50,
    load_best_model_at_end=True,
)


def compute_metrics(y_pred, y_test):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary', zero_division=0
    )
    acc = accuracy_score(y_test, y_pred)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    metric=compute_metrics,
)


trainer.train()
metrics = trainer.evaluate()
print("評価結果:", metrics)


test_texts = [
    "今日はずっと行きたかったカフェに行けて本当に嬉しかった！",
    "締め切りに追われて胃が痛い一日だった。"
]
predictions = model.predict(test_texts)
print("予測結果 (1: 満足, 0: 満足以外):", predictions)

# model.save_pretrained("./setfit-custom-model")