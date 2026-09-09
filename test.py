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