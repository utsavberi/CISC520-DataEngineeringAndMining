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

'''
BOOKED~ C(PSYYR2)+ C(IRSEX)+ C(EDUCCAT2)+ C(IRMARIT)"
+ C(CATAG3)+ C(NEWRACE2)+ C(GOVTPROG)+ C(EMPSTATY)+ C(HVYDRK2)+ C(MJOFLAG)+ C(SUMFLAG)

IVs
most of these were imputed variables:
PSYYR2 - any psychotherapics(Pain reliever(anlyr),  Sedatives(SEDYR), Stimulant(stmyr), Tranqualizers(TRQYR)) use in the last year - 0 no 1 yes
IRSEX - gender - 1 male 2 female
EDUCCAT2 - education - 1: less than high school; 2: high school graduate; 3: some college; 4:college graduate, 5: 12 to 17 years old
IRMARIT - marital status : 1 married, 2 widowed, 3 divorced or seperated, 4 never been married, 99 skip respondent is <14 years old
CATAG3 - age category: 1 12-17 years old, 2 18-25 years old, 3 26-34 years old, 4 35-49 years old, 5 50 years or older
NEWRACE2 - race - 1 white, 2 black, 3 native am, 4 native HI, 5 asian, 6 more than one race, 7 hispanic
GOVTPROG - received any govt assistance(Supplemental Security Income (IRFAMSSI), food stamps (IRFSTAMP), cash assistance (IRFAMPMT), and non-cash assistance (IRFAMSVC)): 1 yes, 2 no
EMPSTATY - employee status - q employed full time, 2 emloyed part time, 3 unemployed, 4 other(incl., not in labour force)
HVYDRK2 -
ALCYR - alcohol past year use - 0 did not use in the past year, 1 used within the past year
MJOFLAG -
MJOYR2 - marijuana past year use  , 0
SUMFLAG
X dont use PAROLREL- on parrole/supervised release from prisonat any time during the past 12 months: 1 yes, 2 no, 85 bad data, 94 dont know 97 refused, 98 blank
X dont use PROBATON - ON probationat any time past 12 month: 1 yes, 2 No, 85 bad data, 94 dont know, 97 refused, 98 blank 
BLNTEVER ever smoked cigar with Marijuana in it 1 yes, 2 no, 4 no logically assigned, 11 yes, 85 bad data, 94 dont know, 97 refused, 98 blank
OTDGNEDL - ever used needlto inject any other drug that was not prescriber - 1 yes, 2 no, 85 bad data, 94 dont kow, 97 refused, 98 blank
RKFQPBLT - wear a seatbelt when ride  front pass seat of car 1 never, 2 seldom, 3 sometimes, 4 always, 85 bad data, 94 dondt know, 97 refused, 98 blank
rsksell - approached by someone selling ill drugs past 30 days 1 yes, 2 no, 85 bad data, 94 dont know, 97 refused, 98 blank
INHOSPYR - stayed overnight as inpt in hosp past 12 mons 1 yes, 2 no, 85 bad data, 94 dont know, 97 refused, 98 blank
ADDERALL - ever used aderall that was not prescribed 1 yes, 2 no, 85 bad data, 94 dont know, 97 refused, 98 blank
COLDMEDS - taken a nonpresc cogh med to get high 1 yes, 2 no, 85 bad data, 94 dont know, 97 refused, 98 blank

IV
any drug
--types of drug
--alcohol

DVs
booked -  ever arrested

--violent non violent
mental health
--depression

'''

df_train = pd.read_csv('data/ICPSR_35509/DS0001/35509-0001-Data.tsv', delimiter='\t',
                       encoding='utf-8')  # shape (55160, 3141)
# df_train = pd.read_csv('data_tail_100.csv')

# feature selection
# print('read done')
# print(df_train.corr()['BOOKED'].sort_values(ascending=False).head(10))
# print(df_train.corr(method='kendall')['BOOKED'].sort_values(ascending=False).head(10))
# print('corr done')

'''
top ten corelations pearson
BOOKED      1.000000
PAROLREL    0.444055
PROBATON    0.385823
BLNTEVER    0.220689
OTDGNEDL    0.194398
RKFQPBLT    0.186551
RSKSELL     0.184227
INHOSPYR    0.169531
ADDERALL    0.152379
COLDMEDS    0.152321

top 10 kendall
BOOKED      1.000000
PAROLREL    0.444055
PROBATON    0.385823
BLNTEVER    0.220689
OTDGNEDL    0.194398
RKFQPBLT    0.186551
RSKSELL     0.184227
INHOSPYR    0.169531
ADDERALL    0.152379
COLDMEDS    0.152321
'''
# df_train.BOOKED.replace([3,85,94,97,98],[1, None, None, None, None], inplace=True)
# df_train.BOOKED.replace([1,2],[0,1], inplace=True)
# df_train.NOBOOKY2.replace([2,3,985,994,997,998,999],[1,1,None,None,None,None,0], inplace=True)
# df_train.BKSRVIOL.replace([85, 89, 94, 97, 98, 99], [None, None, None, None, None, None], inplace=True)
# df_train.BKSRVIOL.replace([3, 2], [1, 0], inplace=True)
df_train.DEPRSYR.replace([-9],[0], inplace=True)
df_train.NEWRACE2.replace([2,3,4,5,6,7], [2,2,2,2,2,2], inplace=True)
df_train.CATAG3.replace([1,5],[None,None], inplace=True)
df_train.dropna(inplace=True)


def run(y, X, df_train):
    modelString = y.upper() + "~"
    X = list(map(lambda x: x.upper(), X))
    modelString = modelString + "C(" + ")+C(".join(X) + ")"
    model = smf.logit(formula=modelString, data=df_train) \
        .fit(maxiter=1000000)

    print(model.summary())
    model_odds = pd.DataFrame(np.exp(model.params), columns=['OR'])
    model_odds['z-value'] = model.pvalues
    model_odds[['2.5%', '97.5%']] = np.exp(model.conf_int())
    print(model_odds)


run('DEPRSYR', ['PSYYR2', 'IRSEX', 'educcat2', 'NEWRACE2', 'GOVTPROG', 'EMPSTATY'], df_train)
# #model
# # "BOOKED~ C(PSYYR2)+ C(IRSEX)+ C(EDUCCAT2)+ C(IRMARIT)+ C(CATAG3)+ C(NEWRACE2)+ C(GOVTPROG)+ C(EMPSTATY)+ C(HVYDRK2)+ C(MJOFLAG)+ C(SUMFLAG)"
# model= smf.logit( formula="BKSRVIOL~ C(PSYYR2)+ C(IRSEX)+ C(EDUCCAT2)+ C(IRMARIT)+ C(CATAG3)+ C(NEWRACE2)+ C(EMPSTATY)+ C(MJOFLAG)+ C(SUMFLAG) ", data= df_train)\
#     .fit(maxiter=1000000)
# # model= smf.logit(formula="BOOKED~ C(PSYYR2)+ C(IRSEX)+ C(EDUCCAT2)+ C(IRMARIT)+ C(CATAG3)+ C(NEWRACE2)+ C(GOVTPROG)+ C(EMPSTATY)+ C(HVYDRK2)+ C(MJOFLAG)+ C(SUMFLAG)", data= df_train).fit()
#
# print(model.summary())
#
# model_odds = pd.DataFrame(np.exp(model.params), columns= ['OR'])
# model_odds['z-value']= model.pvalues
# model_odds[['2.5%', '97.5%']] = np.exp(model.conf_int())
# print(model_odds)


'''
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                 BOOKED   No. Observations:                54965
Model:                          Logit   Df Residuals:                    54963
Method:                           MLE   Df Model:                            1
Date:                Sat, 04 Apr 2020   Pseudo R-squ.:                 0.02000
Time:                        20:58:42   Log-Likelihood:                -22235.
converged:                       True   LL-Null:                       -22688.
Covariance Type:            nonrobust   LLR p-value:                2.448e-199
==================================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept          1.9037      0.013    143.923      0.000       1.878       1.930
C(PSYYR2)[T.1]    -1.1312      0.035    -32.067      0.000      -1.200      -1.062
==================================================================================
                      OR        z-value     2.5%     97.5%
Intercept       6.710979   0.000000e+00  6.53923  6.887239
C(PSYYR2)[T.1]  0.322654  1.264257e-225  0.30110  0.345751

Process finished with exit code 0

                           Logit Regression Results                           
==============================================================================
Dep. Variable:                 BOOKED   No. Observations:                54965
Model:                          Logit   Df Residuals:                    54962
Method:                           MLE   Df Model:                            2
Date:                Sat, 04 Apr 2020   Pseudo R-squ.:                 0.04813
Time:                        21:00:56   Log-Likelihood:                -21596.
converged:                       True   LL-Null:                       -22688.
Covariance Type:            nonrobust   LLR p-value:                     0.000
==================================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept          1.5083      0.016     92.690      0.000       1.476       1.540
C(PSYYR2)[T.1]    -1.1599      0.036    -32.193      0.000      -1.231      -1.089
C(IRSEX)[T.2]      0.8960      0.026     34.805      0.000       0.846       0.946
==================================================================================
                      OR        z-value      2.5%     97.5%
Intercept       4.519101   0.000000e+00  4.377244  4.665555
C(PSYYR2)[T.1]  0.313522  2.246523e-227  0.292146  0.336462
C(IRSEX)[T.2]   2.449841  2.024252e-265  2.329296  2.576624

         Current function value: 0.388970
         Iterations 7
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                 BOOKED   No. Observations:                54965
Model:                          Logit   Df Residuals:                    54956
Method:                           MLE   Df Model:                            8
Date:                Sat, 04 Apr 2020   Pseudo R-squ.:                 0.05767
Time:                        21:04:03   Log-Likelihood:                -21380.
converged:                       True   LL-Null:                       -22688.
Covariance Type:            nonrobust   LLR p-value:                     0.000
====================================================================================
                       coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------------
Intercept            1.5332      0.019     79.031      0.000       1.495       1.571
C(PSYYR2)[T.1]      -1.1571      0.036    -31.898      0.000      -1.228      -1.086
C(IRSEX)[T.2]        0.9054      0.026     35.016      0.000       0.855       0.956
C(NEWRACE2)[T.2]    -0.3629      0.035    -10.304      0.000      -0.432      -0.294
C(NEWRACE2)[T.3]    -0.8907      0.083    -10.773      0.000      -1.053      -0.729
C(NEWRACE2)[T.4]     0.0141      0.170      0.083      0.934      -0.319       0.347
C(NEWRACE2)[T.5]     1.1457      0.100     11.421      0.000       0.949       1.342
C(NEWRACE2)[T.6]    -0.1609      0.063     -2.546      0.011      -0.285      -0.037
C(NEWRACE2)[T.7]     0.0987      0.035      2.824      0.005       0.030       0.167
====================================================================================
                        OR        z-value      2.5%     97.5%
Intercept         4.632976   0.000000e+00  4.460122  4.812529
C(PSYYR2)[T.1]    0.314398  2.858371e-223  0.292821  0.337565
C(IRSEX)[T.2]     2.472972  1.286783e-268  2.350766  2.601531
C(NEWRACE2)[T.2]  0.695641   6.724066e-25  0.649241  0.745356
C(NEWRACE2)[T.3]  0.410379   4.621809e-27  0.348988  0.482570
C(NEWRACE2)[T.4]  1.014174   9.340408e-01  0.726709  1.415351
C(NEWRACE2)[T.5]  3.144519   3.295415e-30  2.583250  3.827735
C(NEWRACE2)[T.6]  0.851370   1.089654e-02  0.752180  0.963640
C(NEWRACE2)[T.7]  1.103750   4.746442e-03  1.030657  1.182027
'''
