import numpy as np
import pandas as pd

arr = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [22, 3, 4],
    [6, 9, 8]
])

missing = [arr > 10]
print(arr)
print(missing)

arrdf = pd.DataFrame(data=arr, columns=['name', 'age', 'sex']);
print(arrdf)
# print([arr>10].index())
# arrdf.drop(missing, inplace=True);
# print (arrdf)
arrdf = arrdf[arrdf < 10].dropna()
print(arrdf)


def cleanData(arrdf):
    return arrdf[arrdf < 10].dropna()
