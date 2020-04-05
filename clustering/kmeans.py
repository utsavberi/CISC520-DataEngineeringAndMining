import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import collections


df_train = pd.read_csv('data/ICPSR_35509/DS0001/35509-0001-Data.tsv', delimiter='\t', encoding='utf-8') # shape (55160, 3141)
# df_train = pd.read_csv('data_tail_5000.csv')
# print(data.shape)
# print(list(data.columns.values)[0:5])  # file header
# print(data.head(100).to_csv('data_head_100.csv'))  # last N rows
# print(np.random.randn(6,4))R
# df = pd.DataFrame(np.random.randn(6,4))
# print(data.columns)
# print (data['CIGEVER'])


print(df_train.shape)
print(df_train.head())
df_train.describe()



x = StandardScaler().fit_transform(df_train)
pca = PCA(n_components = 2, random_state=1)
X_pca = pca.fit_transform(x)
print(pca.explained_variance_ratio_.cumsum()[1])
principalDf = pd.DataFrame(data = X_pca
             , columns = ['principal component 1', 'principal component 2'])
print(principalDf.head())

# print(pca.explained_variance_ratio_.cumsum()[1])

##elbow method
# distortions = []
# K_to_try = range(1, 6)
#
# for i in K_to_try:
#     model = KMeans(
#             n_clusters=i,
#             init='k-means++',
#             n_jobs=-1,
#             random_state=1)
#     model.fit(X_pca)
#     distortions.append(model.inertia_)
# plt.plot(K_to_try, distortions, marker='o')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Distortion')
# plt.show()

#chosing 3 as number of clusters
# use the best K from elbow method
model = KMeans(
    n_clusters=3,
    init='k-means++',
    n_jobs=-1,
    random_state=1)

model = model.fit(X_pca)

y = model.predict(X_pca)

plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], s = 50, c = 'yellow', label = 'Cluster 1')
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], s = 50, c = 'green', label = 'Cluster 2')
plt.scatter(X_pca[y == 2, 0], X_pca[y == 2, 1], s = 50, c = 'red', label = 'Cluster 3')
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s = 100, c = 'blue', label = 'Centroids')
plt.title('Clusters of Participants')
plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')
plt.legend()
plt.grid()
plt.show()

print('K Means Result : ')
print(collections.Counter(y))


# plt.scatter(principalDf['principal component 1'], principalDf['principal component 2'])
# plt.show()