import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import collections
import scipy.cluster.hierarchy as hier #do not remove
from sklearn.cluster import AgglomerativeClustering
import time



print('loading data')
start = time.time()
df_train = pd.read_csv('data/ICPSR_35509/DS0001/35509-0001-Data.tsv', delimiter='\t', encoding='utf-8') # shape (55160, 3141)
# df_train = pd.read_csv('data_tail_1000.csv')
# print(data.shape)
# print(list(data.columns.values)[0:5])  # file header
# print(data.head(100).to_csv('data_head_100.csv'))  # last N rows
# print(np.random.randn(6,4))R
# df = pd.DataFrame(np.random.randn(6,4))
# print(data.columns)
# print (data['CIGEVER'])
end = time.time()
print(end - start)
print('data loaded')


# print(df_train.shape)
# print(df_train.head())
# df_train.describe()


print('scaling data')
start = time.time()
x = StandardScaler().fit_transform(df_train)
end = time.time()
print(end - start)

print('reducing dimenions')
start = time.time()
pca = PCA(n_components = 2, random_state=1)
X_pca = pca.fit_transform(x)
end = time.time()
print(end - start)

# print('creating dendograms')
# dendrogram = hier.dendrogram(hier.linkage(X_pca, method = 'ward'))
# plt.title('Dendrogram')
# plt.xlabel('questions')
# plt.ylabel('Euclidean distances')
# plt.show()

print('creating model')
start = time.time()
model = AgglomerativeClustering(n_clusters = 2, affinity ='euclidean', linkage ='ward')
y = model.fit_predict(X_pca)
end = time.time()
print(end - start)

print('generating plot')
start = time.time()
# generate scatterplot untuk cek hasil grouping (natural/gak)
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], s = 50, c = 'yellow', label = 'Cluster 1')
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], s = 50, c = 'green', label = 'Cluster 2')
plt.title('Clusters of participants')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid()
end = time.time()
print(end - start)
plt.show()

print('AgglomerativeClustering w/ Ward Result : ')
print(collections.Counter(y))



print('done agg')
