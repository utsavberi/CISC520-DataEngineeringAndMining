import pandas as pd
import helper.logitHelper as lh

df_train = pd.read_csv('data/ICPSR_35509/DS0001/35509-0001-Data.tsv', delimiter='\t',
                       encoding='utf-8')  # shape (55160, 3141)

# df_train.BOOKED.replace([3,85,94,97,98],[1, None, None, None, None], inplace=True)
# df_train.BOOKED.replace([1,2],[0,1], inplace=True)
# df_train.NOBOOKY2.replace([2,3,985,994,997,998,999],[1,1,None,None,None,None,0], inplace=True)
df_train.DEPRSYR.replace([-9], [0], inplace=True)
# df_train.BKSRVIOL.replace([85, 89, 94, 97, 98, 99], [None, None, None, None, None, None], inplace=True)
# df_train.BKSRVIOL.replace([3, 2], [1, 0], inplace=True)
df_train.NEWRACE2.replace([2, 3, 4, 5, 6, 7], [2, 2, 2, 2, 2, 2], inplace=True)
df_train.CATAG3.replace([1, 5], [None, None], inplace=True)
df_train.dropna(inplace=True)

lh.runLogit('DEPRSYR', ['MRJYR', 'IRSEX', 'educcat2', 'NEWRACE2', 'GOVTPROG', 'EMPSTATY'], df_train)
