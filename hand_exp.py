# 打印ver2前50行的文字，然后手动评估
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
    nrows=50, 
    usecols=target_columns   # 只保留指定列
)

res1 = []

for index, row in df.iterrows():
    sentence = row['Sentence']
    sadness_score = row['Writer_Sadness']
    
    # 判断是否悲伤
    is_sad = sadness_score > 0

    if is_sad:
        res1.append(1)
    else:
        res1.append(0)

for index, row in df.iterrows():
    sentence = row['Sentence']
    print(f'『{index}』: {sentence}')


# res1 = [1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0]

resme = [1, 0, 1, 1, 0,
 1, 0, 1, 0, 0,
 0, 1, 1, 0, 0,
 1, 0, 0, 0, 1,
 0, 0, 0, 1, 1, 
 0, 0, 0, 0, 0,
 1, 1, 1, 1, 0,
 1, 0, 0, 0, 0,
 0, 1, 0, 0, 1,
 0, 1, 1, 1, 1]
