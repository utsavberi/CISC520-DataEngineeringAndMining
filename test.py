import pandas as pd
import numpy as np

data = pd.read_csv('data/ICPSR_35509/DS0001/35509-0001-Data.tsv', delimiter='\t', encoding='utf-8') # shape (55160, 3141)
# data = pd.read_csv('data_head_100.csv')
# print(data.shape)
# print(list(data.columns.values)[0:5])  # file header
# print(data.head(100).to_csv('data_head_100.csv'))  # last N rows
# print(np.random.randn(6,4))R
# df = pd.DataFrame(np.random.randn(6,4))
# print(data.columns)
# print (data['CIGEVER'])
print(sum(data.isnull().sum()))