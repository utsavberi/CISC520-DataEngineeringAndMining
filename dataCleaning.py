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


arr = np.array([
    [1, 2, 3],
    [4, 5, 625],
    [225, 3, 4],
    [225, 3, 4],
    [225, 97, 4],
    [225, 3, 4],
    [99997, 3, 4],
    [225, 3, 4],
    [225, 3, 4],
    [225, 98, 4],
    [225, 3, 4],
    [225, 3, 4],
    [9988888, 3, 4],
    [6, 295, 99999998]
])


# missing = [arr > 10]
# print(arr)
# print(missing)

df = pd.DataFrame(data=arr, columns=['name', 'age', 'sex']);
print(df)
print(cleanData(df))
# f = (df[df <3].dropna())
# print (f)
# print([arr>10].index())
# arrdf.drop(missing, inplace=True);
# print (df)
# print ('dropped')
# print(np.where((df % 100 != 25) ))
# arrdf =  np.delete(df, df[np.logical_or.reduce((df % 100 != 25) , (df %10!=3), (df %10!=3))])
# print(df)


# ar = np.array([20,44,55,6,7])
# print(ar[np.logical_or.reduce(((arrdf % 100 != 25) , (arrdf %10!=3), (arrdf %10!=3))) ])
