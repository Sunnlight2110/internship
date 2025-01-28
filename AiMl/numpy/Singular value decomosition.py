"""Singular value decomposition is a method decomposing a matrix into smaller matrix """

""" Key uses
    reduce the dimensions of the dataset while retaining important features.
    By approximating a matrix using fewer singular values, we can compress data 
    VD can help remove noise from data by zeroing out smaller singular values.
"""
import numpy as np

np1 = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])
# Perform SVD
U,S,Vt = np.linalg.svd(np1)

"""U will contain the left singular vectors of np1.
S will contain the singular values (in a 1D array, but can be converted into a diagonal matrix).
Vt will be the transpose of the right singular vectors matrix."""

print(f"matrix {U},\n Singular values {S},\n Matrix {Vt}")

# Reconstructing original matrix
sigma = np.diag(S) #Convert singular values into diagonal matrix
print("sigma ",sigma)

renp1 = np.dot(U,np.dot(sigma,Vt))
print("Reconstructed matrix ",renp1)

