from LogisticRegression.abstractLogitRunner import AbstractLogitRunner


class PsychotherapeuticsDrugUseDepression(AbstractLogitRunner):
    def __init__(self, dataPath):
        super().__init__('DEPRSYR', ['PSYYR2', 'IRSEX', 'educcat2', 'NEWRACE2', 'GOVTPROG', 'EMPSTATY'], dataPath)

    def _cleanData(self):
        self._data.DEPRSYR.replace([-9], [0], inplace=True)
        super()._cleanData()


class AnyIllicitDrugUseDepression(AbstractLogitRunner):
    def __init__(self, dataPath):
        super().__init__('DEPRSYR', ['SUMYR', 'IRSEX', 'educcat2', 'NEWRACE2', 'GOVTPROG', 'EMPSTATY'], dataPath)

    def _cleanData(self):
        self._data.DEPRSYR.replace([-9], [0], inplace=True)
        super()._cleanData()


class MarijuanaUseDepression(AbstractLogitRunner):
    def __init__(self, dataPath):
        super().__init__('DEPRSYR', ['MRJYR', 'IRSEX', 'educcat2', 'NEWRACE2', 'GOVTPROG', 'EMPSTATY'], dataPath)

    def _cleanData(self):
        self._data.DEPRSYR.replace([-9], [0], inplace=True)
        super()._cleanData()


class PsychotherapeuticsDrugUseAnyArrest(AbstractLogitRunner):
    def __init__(self, dataPath):
        super().__init__('NOBOOKY2', ['PSYYR2', 'IRSEX', 'educcat2', 'NEWRACE2', 'GOVTPROG', 'EMPSTATY'], dataPath)

    def _cleanData(self):
        self._data.NOBOOKY2.replace([2, 3, 985, 994, 997, 998, 999], [1, 1, None, None, None, None, 0], inplace=True)
        super()._cleanData()


class AnyIllicitDrugUseAnyArrest(AbstractLogitRunner):
    def __init__(self, dataPath):
        super().__init__('NOBOOKY2', ['SUMYR', 'IRSEX', 'educcat2', 'NEWRACE2', 'GOVTPROG', 'EMPSTATY'], dataPath)

    def _cleanData(self):
        self._data.NOBOOKY2.replace([2, 3, 985, 994, 997, 998, 999], [1, 1, None, None, None, None, 0], inplace=True)
        super()._cleanData()


class MarijuanaUseAnyArrest(AbstractLogitRunner):
    def __init__(self, dataPath):
        super().__init__('NOBOOKY2', ['MRJYR', 'IRSEX', 'educcat2', 'NEWRACE2', 'GOVTPROG', 'EMPSTATY'], dataPath)

    def _cleanData(self):
        self._data.NOBOOKY2.replace([2, 3, 985, 994, 997, 998, 999], [1, 1, None, None, None, None, 0], inplace=True)
        super()._cleanData()


class PsychotherapeuticsDrugUseBookedForViolent(AbstractLogitRunner):
    def __init__(self, dataPath):
        super().__init__('BKSRVIOL', ['PSYYR2', 'IRSEX', 'educcat2', 'NEWRACE2', 'GOVTPROG', 'EMPSTATY'], dataPath)

    def _cleanData(self):
        self._data.BKSRVIOL.replace([85, 89, 94, 97, 98, 99], [None, None, None, None, None, None], inplace=True)
        self._data.BKSRVIOL.replace([3, 2], [1, 0], inplace=True)
        super()._cleanData()


class AnyIllicitDrugUseBookedForViolent(AbstractLogitRunner):
    def __init__(self, dataPath):
        super().__init__('BKSRVIOL', ['SUMYR', 'IRSEX', 'educcat2', 'NEWRACE2', 'GOVTPROG', 'EMPSTATY'], dataPath)

    def _cleanData(self):
        self._data.BKSRVIOL.replace([85, 89, 94, 97, 98, 99], [None, None, None, None, None, None], inplace=True)
        self._data.BKSRVIOL.replace([3, 2], [1, 0], inplace=True)
        super()._cleanData()


class MarijuanaUseBookedForViolent(AbstractLogitRunner):
    def __init__(self, dataPath):
        super().__init__('BKSRVIOL', ['MRJYR', 'IRSEX', 'educcat2', 'NEWRACE2', 'GOVTPROG', 'EMPSTATY'], dataPath)

    def _cleanData(self):
        self._data.BKSRVIOL.replace([85, 89, 94, 97, 98, 99], [None, None, None, None, None, None], inplace=True)
        self._data.BKSRVIOL.replace([3, 2], [1, 0], inplace=True)
        super()._cleanData()


dataPath = '../data/ICPSR_35509/DS0001/35509-0001-Data.tsv'
runnables = [
    AnyIllicitDrugUseAnyArrest,
    PsychotherapeuticsDrugUseAnyArrest,
    MarijuanaUseAnyArrest,
    AnyIllicitDrugUseBookedForViolent,
    PsychotherapeuticsDrugUseBookedForViolent,
    MarijuanaUseBookedForViolent,
    PsychotherapeuticsDrugUseDepression,
    AnyIllicitDrugUseDepression,
    MarijuanaUseDepression,
]

for algo in runnables:
    algo = algo(dataPath)
    algo.run()
