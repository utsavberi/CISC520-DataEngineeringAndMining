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
import statsmodels.api as sm
from scipy import stats

df_train = pd.read_csv('data/ICPSR_35509/DS0001/35509-0001-Data.tsv', delimiter='\t', encoding='utf-8') # shape (55160, 3141)
# df_train = pd.read_csv('data_tail_5000.csv')
df_train.BOOKED.replace([3,85,94,97,98],[1, None, None, None, None], inplace=True)
# df_train.BKLARCNY.replace([3,85,94,97,98, 99, 89],[1, None, None, None, None, None, None], inplace=True)

df_train.dropna(inplace=True)

X = df_train[['PSYYR2', 'IRSEX', 'EDUCCAT2', 'IRMARIT', 'CATAG3', 'NEWRACE2', 'GOVTPROG', 'EMPSTATY', 'HVYDRK2', 'MJOFLAG', 'SUMFLAG']]
y = df_train['BOOKED']

clf = LogisticRegression(random_state=0)
clf.fit(X, y)
print('odds ratio')
print(clf.classes_)
print(np.exp(clf.coef_)) #odds ratio
print(clf.coef_) #relationship

print('method 2')

# logit only accepts 0/1 as target values
df_train.BOOKED.replace([1,2],[0,1], inplace=True)
est = sm.Logit(y, X)
est2 = est.fit()
print(est2.summary())

params = est2.params
conf = est2.conf_int()
conf['Odds Ratio'] = params
conf.columns = ['5%', '95%', 'Odds Ratio']
print(np.exp(conf))