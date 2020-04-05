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

    )].dropna().drop(['Unnamed: 0', 'CASEID','QUESTID2'], axis=1, errors='ignore')

#
df_train = pd.read_csv('data_tail_5000.csv')
np.savetxt('column_names.csv',X=np.array(df_train.columns), fmt="%s");
# print(df_train)
# print(df_train.columns.shape)
# a = np.arange(0,3142*2)
# a=np.reshape(a,(3142,-1))
# print(a.shape)
# print(df_train.columns)
# b = np.array(df_train.columns)
# print(type(b))
# print(b.shape)
# print(b[:,None].shape)
# print(b.T.shape)
# print(np.hstack((a, df_train.columns[:,None])))
# df_train=df_train.drop(['Unnamed: 0', 'CASEID','QUESTID2'], axis=1)
# print(df_train)
# df_train = pd.read_csv('data_tail_5000.csv')
# print(df_train.columns.shape)
# print(np.array(df_train.columns).shape)