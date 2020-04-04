'''
a = Number of exposed cases

b = Number of exposed non-cases

c = Number of unexposed cases

d = Number of unexposed non-cases

OR=a/cb/d=adbc
OR=(n)exposedcases/(n)unexposedcases(n)exposednon-cases/(n)unexposednon-cases=(n)exposedcases×(n)unexposednon-cases(n)exposednon-cases×(n)unexposedcases

'''

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
import statsmodels.formula.api as smf

# df_train = pd.read_csv('data/ICPSR_35509/DS0001/35509-0001-Data.tsv', delimiter='\t',
#                        encoding='utf-8')  # shape (55160, 3141)
df_train = pd.read_csv('data_tail_5000.csv')
df_train.BOOKED.replace([3, 85, 94, 97, 98], [1, None, None, None, None], inplace=True)

sex_booked = df_train[['IRSEX', 'BOOKED']]

# plus_plus = sex_booked[((sex_booked['IRSEX'] == 1) & (sex_booked['BOOKED'] == 1))]
# print(plus_plus.shape)
#
# plus_minus = sex_booked[((sex_booked['IRSEX'] == 1) & (sex_booked['BOOKED'] == 2))]
# print(plus_minus.shape)
#
# minus_plus = sex_booked[((sex_booked['IRSEX'] == 2) & (sex_booked['BOOKED'] == 1))]
# print(minus_plus.shape)
#
# minus_minus = sex_booked[((sex_booked['IRSEX'] == 2) & (sex_booked['BOOKED'] == 2))]
# print(minus_minus.shape)
# '''tail 5000
# (458, 2)
# (1959, 2)
# (197, 2)
# (2328, 2)
# '''

crosstab = pd.crosstab(sex_booked['IRSEX'], sex_booked['BOOKED'], margins=False)
print(crosstab)
a = crosstab.iloc[0, 0]
b = crosstab.iloc[0, 1]
c = crosstab.iloc[1, 0]
d = crosstab.iloc[1, 1]

odds = (a * d) / (b * c)
print('odd ratio %f', odds)

crosstab = pd.crosstab(df_train['MJOFLAG'], sex_booked['BOOKED'], margins=False)
print(crosstab)
a = crosstab.iloc[0, 0]
b = crosstab.iloc[0, 1]
c = crosstab.iloc[1, 0]
d = crosstab.iloc[1, 1]

odds = (a * d) / (b * c)
print('odd ratio %f', odds)

crosstab = pd.crosstab(df_train['SUMFLAG'], sex_booked['BOOKED'], margins=False)
print(crosstab)
a = crosstab.iloc[0, 0]
b = crosstab.iloc[0, 1]
c = crosstab.iloc[1, 0]
d = crosstab.iloc[1, 1]

odds = (a * d) / (b * c)
print('odd ratio %f', odds)

crosstab = pd.crosstab(df_train['PSYYR2'], sex_booked['BOOKED'], margins=False)
print(crosstab)
a = crosstab.iloc[0, 0]
b = crosstab.iloc[0, 1]
c = crosstab.iloc[1, 0]
d = crosstab.iloc[1, 1]

odds = (a * d) / (b * c)
print('odd ratio %f', odds)

crosstab = pd.crosstab(df_train['GOVTPROG'], sex_booked['BOOKED'], margins=False)
print(crosstab)
a = crosstab.iloc[0, 0]
b = crosstab.iloc[0, 1]
c = crosstab.iloc[1, 0]
d = crosstab.iloc[1, 1]

odds = (a * d) / (b * c)
print('odd ratio %f', odds)

crosstab = pd.crosstab(df_train['HVYDRK2'], sex_booked['BOOKED'], margins=False)
print(crosstab)
a = crosstab.iloc[0, 0]
b = crosstab.iloc[0, 1]
c = crosstab.iloc[1, 0]
d = crosstab.iloc[1, 1]

odds = (a * d) / (b * c)
print('odd ratio %f', odds)

crosstab = pd.crosstab(df_train['SUMYR'], sex_booked['BOOKED'], margins=False)
print(crosstab)
a = crosstab.iloc[0, 0]
b = crosstab.iloc[0, 1]
c = crosstab.iloc[1, 0]
d = crosstab.iloc[1, 1]

odds = (a * d) / (b * c)
print('odd ratio %f', odds)

df_train['SUMYR'].replace([0, 1], [1, 0], inplace=True)
crosstab = pd.crosstab(df_train['SUMYR'], sex_booked['BOOKED'], margins=False)
print(crosstab)
a = crosstab.iloc[0, 0]
b = crosstab.iloc[0, 1]
c = crosstab.iloc[1, 0]
d = crosstab.iloc[1, 1]

odds = (a * d) / (b * c)
print('odd ratio %f', odds)

df_train['PSYYR2'].replace([0, 1], [1, 0], inplace=True)
crosstab = pd.crosstab(df_train['PSYYR2'], sex_booked['BOOKED'], margins=False)
print(crosstab)
a = crosstab.iloc[0, 0]
b = crosstab.iloc[0, 1]
c = crosstab.iloc[1, 0]
d = crosstab.iloc[1, 1]

odds = (a * d) / (b * c)
print('odd ratio %f', odds)

x = np.array([[1, 1],
          [1, 1],
          [1, 1],
          [1, 1],
          [1, 1],
          [1, 1],
          [1, 1],
          [1, 0],
          [1, 0],
          [1, 0],

          [0, 1],
          [0, 1],
          [0, 1],
          [0, 0],
          [0, 0],
          [0, 0],
          [0, 0],
          [0, 0],
          [0, 0],
          [0, 0]]).astype(np.float)
          #1 is male 0 is female
test = pd.DataFrame(data = x, index = np.arange(0, 20), columns=['Gender', 'Admission'] );
crosstab = pd.crosstab(test['Gender'], test['Admission'], margins=False)
print('crr')
print(crosstab)
a = crosstab.iloc[1, 1]
b = crosstab.iloc[1, 0]
c = crosstab.iloc[0, 1]
d = crosstab.iloc[0, 0]

print(a)
print(b)
print(c)
print(d)
odds = (a * d) / (b * c)
print('odd ratio of makes %f', odds)
# print(test)


'''
alternate https://pythonfordatascience.org/logistic-regression-python/
'''
model= smf.logit(formula="Admission~ C(Gender)", data= test).fit()
model.summary()

model_odds = pd.DataFrame(np.exp(model.params), columns= ['OR'])
model_odds['z-value']= model.pvalues
model_odds[['2.5%', '97.5%']] = np.exp(model.conf_int())
print(model_odds)

# clf = LogisticRegression(random_state=0)
# clf.fit(np.array(test['Gender']).reshape(-1,1), test['Admission'])
# print('odds ratio')
# print(clf.classes_)
# print(np.exp(clf.coef_)) #odds ratio
# print(clf.coef_) #relationship

# est = sm.Logit(test['Admission'], ['Gender'],)
# est2 = est.fit()
# print(est2.summary())
#
# params = est2.params
# conf = est2.conf_int()
# conf['Odds Ratio'] = params
# conf.columns = ['5%', '95%', 'Odds Ratio']
# print(np.exp(conf))
