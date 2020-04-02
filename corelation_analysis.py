import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import collections
from scipy.stats import kendalltau
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

# df_train = pd.read_csv('data/ICPSR_35509/DS0001/35509-0001-Data.tsv', delimiter='\t', encoding='utf-8') # shape (55160, 3141)
df_train = pd.read_csv('data_tail_5000.csv')

#cleaning
df_train.BOOKED.replace([3,85,94,97,98],[1, None, None, None, None], inplace=True)
df_train.dropna(inplace=True)

# print(df_train.columns)
##column
##male/female=>IRSEX
##arrested or booked for crime=>BOOKED
# print(df_train['IRSEX'])
# print(df_train['BOOKED'].unique())
# print(df_train[['IRSEX','BOOKED']].corr(method='kendall'))

print(kendalltau(df_train['IRSEX'], df_train['BOOKED']))
clf = LogisticRegression(random_state=0)
clf.fit(df_train[['IRSEX']], df_train['BOOKED'])
# print(np.exp(clf.coef_))
# print(clf.intercept_)
print(clf.coef_)

# clf = LinearRegression()
# clf.fit(df_train[['IRSEX']], df_train['BOOKED'])
# print(np.exp(clf.coef_))
# print(clf.intercept_)
# print(clf.coef_)