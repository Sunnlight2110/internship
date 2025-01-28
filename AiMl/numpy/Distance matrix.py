"""
A distance matrix is a square matrix in which each element represents the distance between a pair of points in a dataset.
"""
import numpy as np
from scipy.spatial.distance import cdist

"""Euclidean Distance
Euclidean distance between two points(rows) in d-dimensional space"""

np1 = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9],
])

distance = cdist(np1,np1,metric='euclidean') # Default is euclidean
print("Distance ",distance)

distance = cdist(np1,np1,metric='cityblock') # Manhattan distance
print("Distance ",distance)

distance = cdist(np1,np1,metric='Minkowski', p=3) # Minkowski  distance, p=1: manhattan dis, p=2 euclidean distance
print("Distance ",distance)

