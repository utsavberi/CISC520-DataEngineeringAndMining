import numpy as np
import pandas as pd


def cleanData(arrdf):
    # print([((arrdf != 4) & (arrdf!=225))])
    return arrdf[(
            # (arrdf % 100 != 93)
            # &
            # (arrdf % 100 != 94)
            # &
            (arrdf % 100 != 97)
            # & (arrdf % 100 != 98)
            # & (arrdf % 100 != 99)
            # & (arrdf % 100 != 83)
            & (arrdf % 100 != 85)

    )].dropna()
