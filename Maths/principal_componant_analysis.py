import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA   #Is Principal component analysis

"""• Simple machine learning algorithm
Unsupervised: enables identification of structure in unlabeled data
Like eigendecomposition and SVD, enables lossy compression
 To minimize both loss of precision and data footprint, first principal component contains most variance (data structure), 
    second PC contains next most, and so on
 Involves many linear algebra concepts already covered, e.g.:
    Norms
    Orthogonal and identity matrices
    Trace operator
"""

# Iris is flower dataset
iris = datasets.load_iris()
print(iris.get('feature_names'))

pca = PCA(n_components=2)   #2 comp because we need x,y plot
X = pca.fit_transform(iris.data)    #calculates fit components of data and applies data
print(X.shape)  #same numbers of rows, but instead of 4 cols, have 2 principal components

# _= plt.scatter(X[:,0],X[:,1])

print(iris.target.shape)
print(iris.target[:])   #Contains target values(labels) associated with each data points(species)

plt.scatter(X[:,0],X[:,1],c=iris.target)
plt.show()