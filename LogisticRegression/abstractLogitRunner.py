import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
from tabulate import tabulate

pdtabulate = lambda df: tabulate(df, headers='keys', tablefmt='psql')


class AbstractLogitRunner:
    def __init__(self, y: str, X: list, dataPath):
        self.dataPath = dataPath
        self.y = y
        self.X = X
        self._data = None
        pd.options.display.max_columns = None
        pd.options.display.width = None

    def run(self):
        self.__loadData()
        self._cleanData()
        self.__runLogit()

    def __loadData(self):
        self._data = pd.read_csv(self.dataPath, delimiter='\t', encoding='utf-8')

    def _cleanData(self):
        self._data.NEWRACE2.replace([2, 3, 4, 5, 6, 7], [2, 2, 2, 2, 2, 2], inplace=True)
        self._data.CATAG3.replace([1, 5], [None, None], inplace=True)
        self._data.dropna(inplace=True)

    def __runLogit(self):
        modelString = self.y.upper() + "~"
        X = list(map(lambda x: x.upper(), self.X))
        modelString = modelString + "C(" + ")+C(".join(X) + ")"
        print(modelString)

        model = smf.logit(formula=modelString, data=self._data) \
            .fit(maxiter=1000000)

        self.displayResults(model)

    def displayResults(self, model):
        # print(model.summary())
        model_odds = pd.DataFrame(np.exp(model.params), columns=['OR'])
        model_odds['coef'] = model.params
        model_odds['p'] = model.pvalues
        model_odds['signi'] = model.pvalues
        model_odds.loc[model_odds['p'] < 0.05, 'signi'] = '*'
        model_odds.loc[model_odds['p'] < 0.01, 'signi'] = '**'
        model_odds.loc[model_odds['p'] < 0.001, 'signi'] = '***'
        model_odds.loc[model_odds['p'] > 0.05, 'signi'] = ''
        model_odds[['2.5%', '97.5%']] = np.exp(model.conf_int())

        self.renameIndexValues(model_odds)

        print(pdtabulate(model_odds))
        print('======================================================\n\n')

    def renameIndexValues(self, model_odds):
        mapping = {
            'C(PSYYR2)[T.1]': 'PsychotherapeuticsPastYearUse:Yes',
            'C(IRSEX)[T.2]': 'Gender:female',
            'C(EDUCCAT2)[T.2]': 'Education:HighSchoolGraduate',
            'C(EDUCCAT2)[T.3]': 'Education:SomeCollege',
            'C(EDUCCAT2)[T.4]': 'Education:CollegeGraduate',
            'C(NEWRACE2)[T.2]': 'Race:NonWhite',
            'C(GOVTPROG)[T.2]': 'RcvdGovtAssistance:No',
            'C(EMPSTATY)[T.2]': 'EmploymentStatus:PartTime',
            'C(EMPSTATY)[T.3]': 'EmploymentStatus:Unemployed',
            'C(EMPSTATY)[T.4]': 'EmploymentStatus:Other(notInLbrForce)',
            'C(SUMYR)[T.1]': 'AnyIllicitDrugPastYearUse:Yes',
            'C(MRJYR)[T.1]': 'MarijuanaPastYearUse:Yes'
        }
        as_list = model_odds.index.tolist()

        for elem in as_list:
            if (elem in mapping):
                as_list[as_list.index(elem)] = mapping[elem]
        model_odds.index = as_list
