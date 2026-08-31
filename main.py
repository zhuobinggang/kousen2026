import pandas as pd

# 1. 定义需要保留的列名（Sentence + 9 个 Writer 情感列）
target_columns = [
    'Sentence',
    'Writer_Joy',
    'Writer_Sadness',
    'Writer_Anticipation',
    'Writer_Surprise',
    'Writer_Anger',
    'Writer_Fear',
    'Writer_Disgust',
    'Writer_Trust',
    'Writer_Sentiment'
]

# 2. 读取文件
df = pd.read_csv(
    './wrime/wrime-ver2.tsv',
    sep='\t',
    nrows=100,              # 只读取前 100 行数据
    usecols=target_columns   # 只保留指定列
)

# 3. 查看结果
print(df.head())


sadness_count = (df['Writer_Sadness'] > 0).sum()
print(f"Writer_Sadness 存在的行数（> 0）: {sadness_count}")
# > Writer_Sadness 存在的行数（> 0）: 50

