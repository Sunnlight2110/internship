import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

customers_df = pd.read_csv('C:\\Users\\My PC\\Desktop\\Internship\\Machine Learning (Codes and Data Files)\\Data\\Income Data.csv')
# print(customers_df.head())
# print(customers_df.info())

# sns.lmplot(
#     x="age",y="income",
#     data=customers_df,
#     fit_reg=False,
#     height=4
# )
# plt.show()

clusters = KMeans(3)
clusters.fit(customers_df)

# KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300, 
#  n_clusters=3, n_init=10, n_jobs=None, precompute_distances='auto', 
#  random_state=None, tol=0.0001, verbose=0)
"""
KMeans clustering algorithm for unsupervised learning.

This implementation of the KMeans algorithm will try to partition the data into
the specified number of clusters using a variety of parameters for optimization.

Parameters:
- algorithm: {'auto', 'full', 'elkan'}, default='auto'
    - The KMeans algorithm to use. 'auto' will choose the best algorithm based on the dataset.
    - 'full' is the standard KMeans algorithm, while 'elkan' is an optimized version (faster for dense datasets).
  
- copy_x: bool, default=True
    - If True, the input data is copied; if False, the data is modified in-place.

- init: {'k-means++', 'random'}, default='k-means++'
    - Method for initialization. 'k-means++' spreads out the initial cluster centers, leading to faster convergence.
    - 'random' randomly selects the initial cluster centers.

- max_iter: int, default=300
    - Maximum number of iterations to run the KMeans algorithm.

- n_clusters: int, default=3
    - The number of clusters to form and the number of centroids to generate.

- n_init: int, default=10
    - The number of times the algorithm will be run with different initializations.
    - The final results will be the best output of these runs.

- n_jobs: int or None, default=None
    - The number of CPUs to use for the computation. If None, it uses 1 core. Set to -1 to use all available cores.

- precompute_distances: {'auto', True, False}, default='auto'
    - Whether to precompute distances between points. This can speed up the computation in some cases.

- random_state: int, RandomState instance, or None, default=None
    - The seed for random number generation, which ensures reproducibility.

- tol: float, default=0.0001
    - The tolerance to declare convergence. If the change in cluster centers is less than this value, the algorithm stops.

- verbose: int, default=0
    - Verbosity mode. Set to 1 to print progress during fitting.

"""

customers_df['clusterid'] = clusters.labels_
# print(customers_df[:5])

markers = ['+','*','^']

# sns.lmplot(
#     x='age',y='income',
#     data=customers_df,
#     hue='clusterid',
#     fit_reg=False,
#     markers=markers,
#     height=4
# )
# plt.show()

scaler = StandardScaler()
scaled_customer_df = scaler.fit_transform(
    customers_df[['age','income']]
)
# print(scaled_customer_df)

clusters_new = KMeans(3,random_state=42)
clusters_new.fit(scaled_customer_df)
customers_df['clusterid_new'] = clusters_new.labels_
# sns.lmplot(
#     x='age',y='income',
#     data=customers_df,
#     hue='clusterid_new',
#     fit_reg=False,
#     markers=markers,
#     height=4
# )
# plt.show()

# print(clusters.cluster_centers_)
# print(customers_df.groupby('clusterid')[['age','income']].agg(['mean','std']).reset_index())

# ===================================================================   beer df
beer_df = pd.read_csv('C:\\Users\\My PC\\Desktop\\Internship\\Machine Learning (Codes and Data Files)\\Data\\beer.csv')
# print(beer_df)
scaled_customer_df = scaler.fit_transform(beer_df[['calories','sodium','alcohol','cost']])

# drendrogram to visualize
# camp = sns.cubehelix_palette(
#     as_cmap=True, rot= -.3, light=1
# )
# sns.clustermap(
#     scaled_customer_df,
#     cmap=camp, linewidths=.2,
#     figsize=(8,8)
# )
# plt.show()
# print(beer_df.iloc[[10,16]])

