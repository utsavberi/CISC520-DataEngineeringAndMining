import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def runLogit(y, X, df_train):
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
