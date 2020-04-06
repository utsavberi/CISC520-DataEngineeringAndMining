import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from clustering.dataCleaning import cleanData


df_train = pd.read_csv('data/ICPSR_35509/DS0001/35509-0001-Data.tsv', delimiter='\t', encoding='utf-8') # shape (55160, 3141)
# df_train = pd.read_csv('data_tail_5000.csv')
df_train = cleanData(df_train)
df_train_columns = np.array(df_train.columns)
print(df_train.shape)
print(df_train_columns)
# i = 0
# for col in df_train.columns:
#     print(i)
#     print(col)
#     i = i+1
df_train = StandardScaler().fit_transform(df_train)
# print(df_train.columns)
clusters = np.genfromtxt('pca_data_no_missing_full_clusters', delimiter=',')
# clusters = clusters[-5000:]
print(clusters.shape)
# df_train['clusters'] = clusters
# print(df_train.shape)
# print(df_train[0, :])
# print(clusters)
# print(clusters.T)
results = []

for column in df_train.T:
    # print('here')
    # print(column.shape)
    # print(column[clusters == 1].mean())
    # print(column[clusters == 0].mean())
    results.append([column[clusters == 0].mean(),column[clusters == 1].mean()])
    # print(column)

results = np.array(results)

# np.savetxt('column_means_w_cluster_id', np.hstack((df_train_columns, results.shape)))

labels = np.arange(0, results.shape[0])
# print(labels.shape)
cluster1_means = results[:, 0]
cluster2_means = results[:, 1]
# np.savetxt('cluster1_column_means', X=cluster1_means)
# np.savetxt('cluster2_column_means', X=cluster2_means)
# print(results.shape)
# print(df_train_columns.shape)
# np.savetxt('column names.csv', X=np.array(df_train_columns, dtype='string'), delimiter=',')
# np.savetxt('gaussian_cluster_analysis_no_missing_data.csv', X=np_array, delimiter=',')
# np_array = np.array(np.hstack((np.array(df_train_columns), results)))
# np.savetxt('gaussian_cluster_analysis_no_missing_data_w_columns.csv', X=np_array, delimiter=',', fmt="%s")

# print(cluster1_means.shape)
# print(cluster2_means.shape)



width = 1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(labels - width/2, cluster1_means, width, label='Cluster1')
rects2 = ax.bar(labels + width/2, cluster2_means, width, label='Cluster2')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
# ax.set_xticks(labels)
# ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


# autolabel(rects1)
# autolabel(rects2)

fig.tight_layout()
print('showing')
plt.show()
print('done')