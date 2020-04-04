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
import statsmodels.formula.api as smf

from scipy import stats

df_train = pd.read_csv('data/ICPSR_35509/DS0001/35509-0001-Data.tsv', delimiter='\t', encoding='utf-8') # shape (55160, 3141)
# df_train = pd.read_csv('data_tail_5000.csv')
df_train.BOOKED.replace([3,85,94,97,98],[1, None, None, None, None], inplace=True)
# df_train.BKLARCNY.replace([3,85,94,97,98, 99, 89],[1, None, None, None, None, None, None], inplace=True)
df_train['PSYYR2'].replace([0,1],[1,0], inplace=True)

df_train.dropna(inplace=True)

#irsex 1 is male 2 is female
#booked 1 is yes 2 is no
X = df_train[['PSYYR2', 'IRSEX', 'EDUCCAT2', 'IRMARIT', 'CATAG3', 'NEWRACE2', 'GOVTPROG', 'EMPSTATY', 'HVYDRK2', 'MJOFLAG', 'SUMFLAG']]
y = df_train['BOOKED']

# clf = LogisticRegression(random_state=0)
# clf.fit(X, y)
# print('odds ratio')
# print(clf.classes_)
# print(np.exp(clf.coef_)) #odds ratio
# print(clf.coef_) #relationship

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


'''
alternate https://pythonfordatascience.org/logistic-regression-python/
'''
model= smf.logit(formula="BOOKED~ C(PSYYR2)+ C(IRSEX)+ C(EDUCCAT2)+ C(IRMARIT)+ C(CATAG3)+ C(NEWRACE2)+ C(GOVTPROG)+ C(EMPSTATY)+ C(HVYDRK2)+ C(MJOFLAG)+ C(SUMFLAG)", data= df_train).fit()
model.summary()

model_odds = pd.DataFrame(np.exp(model.params), columns= ['OR'])
model_odds['z-value']= model.pvalues
model_odds[['2.5%', '97.5%']] = np.exp(model.conf_int())
print(model_odds)


'''
output
odds ratio
[1. 2.]
[[0.81313405 2.54305379 1.29027551 0.94548162 0.83498834 1.01433999
  2.10557529 1.06417539 0.65472256 1.85503575 0.17681823]]
[[-0.2068593   0.93336564  0.25485577 -0.05606083 -0.18033752  0.01423815
   0.74458873  0.06220022 -0.42354371  0.61790397 -1.73263302]]
method 2
Optimization terminated successfully.
         Current function value: 0.331059
         Iterations 7
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                 BOOKED   No. Observations:                54965
Model:                          Logit   Df Residuals:                    54954
Method:                           MLE   Df Model:                           10
Date:                Fri, 03 Apr 2020   Pseudo R-squ.:                  0.1980
Time:                        23:33:46   Log-Likelihood:                -18197.
converged:                       True   LL-Null:                       -22688.
Covariance Type:            nonrobust   LLR p-value:                     0.000
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
PSYYR2        -0.2088      0.042     -4.966      0.000      -0.291      -0.126
IRSEX          0.9089      0.024     37.793      0.000       0.862       0.956
EDUCCAT2       0.2467      0.011     21.864      0.000       0.225       0.269
IRMARIT       -0.0596      0.009     -6.724      0.000      -0.077      -0.042
CATAG3        -0.1913      0.011    -17.132      0.000      -0.213      -0.169
NEWRACE2       0.0094      0.006      1.683      0.092      -0.002       0.020
GOVTPROG       0.7122      0.026     26.933      0.000       0.660       0.764
EMPSTATY       0.0654      0.009      7.477      0.000       0.048       0.083
HVYDRK2       -0.4384      0.042    -10.455      0.000      -0.521      -0.356
MJOFLAG        0.6314      0.035     17.909      0.000       0.562       0.700
SUMFLAG       -1.7546      0.035    -50.186      0.000      -1.823      -1.686
==============================================================================
                5%       95%  Odds Ratio
PSYYR2    0.747384  0.881289    0.811579
IRSEX     2.367366  2.601407    2.481629
EDUCCAT2  1.251785  1.308391    1.279775
IRMARIT   0.925951  0.958671    0.942169
CATAG3    0.807982  0.844139    0.825862
NEWRACE2  0.998456  1.020564    1.009450
GOVTPROG  1.935499  2.146894    2.038458
EMPSTATY  1.049449  1.086065    1.067600
HVYDRK2   0.594136  0.700296    0.645036
MJOFLAG   1.754651  2.014692    1.880181
SUMFLAG   0.161523  0.185248    0.172980

Process finished with exit code 0

'''