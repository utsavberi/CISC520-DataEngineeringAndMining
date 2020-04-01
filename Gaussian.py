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
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from dataCleaning import cleanData


'''
print('loading data')
start = time.time()
df_train = pd.read_csv('data/ICPSR_35509/DS0001/35509-0001-Data.tsv', delimiter='\t', encoding='utf-8') # shape (55160, 3141)
# df_train = pd.read_csv('data_tail_1000.csv')
end = time.time()
print(end - start)
print('data loaded')


df_train=cleanData(df_train)

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

np.savetxt("pca_data_full_no_missing_data.csv", X_pca, delimiter=",")

exit(0)
'''
start = time.time()

X_pca = np.genfromtxt('pca_data_full_no_missing_data.csv', delimiter=',') # shape (55160, 3141)
end = time.time()
print(end - start)
# print('loading data')
# start = time.time()
# df_train = pd.read_csv('data/ICPSR_35509/DS0001/35509-0001-Data.tsv', delimiter='\t', encoding='utf-8') # shape (55160, 3141)
# df_train = pd.read_csv('data_tail_1000.csv')

# end = time.time()
# print(end - start)
# print('data loaded')
# print(df_train.shape)
# df_train=cleanData(df_train)
# print(df_train.shape)
#
# print('scaling data')
# start = time.time()
# x = StandardScaler().fit_transform(df_train)
# end = time.time()
# print(end - start)
#
# print('reducing dimenions')
# start = time.time()
# pca = PCA(n_components = 2, random_state=1)
# X_pca = pca.fit_transform(x)
# end = time.time()
# print(end - start)

print('creating model')
start = time.time()

gmm = GaussianMixture(n_components=2)
gmm.fit(X_pca)
y = gmm.predict(X_pca)
np.savetxt('pca_data_no_missing_full_clusters', y)
#
end = time.time()
print(end - start)
print('generating plot')
start = time.time()
# generate scatterplot untuk cek hasil grouping (natural/gak)
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], s = 50, c = 'yellow', label = 'Cluster 1')
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], s = 50, c = 'red', label = 'Cluster 3')
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
