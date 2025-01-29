import numpy as np

"""
As on slides, SVD of matrix A is:

A = UDVT

Where:

U is an orthogonal m X m matrix; its columns are the left-singular vectors of A. crossponds rows of output.
V is an orthogonal n x n matrix; its columns are the right-singular vectors of A. crossponds cols of output.
D is a diagonal m X n matrix; elements along its diagonal are the singular values of A. crossponds direction of output.
"""

"""Assumptions:
    Any Matrix: Works for any matrix A of size m × n (rectangular or square).
    Decomposability: Always decomposes into U, Σ, and V^T.
    Rank: The number of non-zero singular values is the rank of A.
    Orthogonality: U and V are orthogonal matrices (their columns are orthonormal).
    Singular Values: Non-negative, sorted in descending order, and represent the matrix's strength in each direction."""

"""SVD is one of the most powerful tools in linear algebra, used for:

    Dimensionality reduction
    Data compression
    Solving least-squares problems
    Principal Component Analysis (PCA)
"""

A = np.array([
    [-1,2],
    [3,-2],
    [5,7]
])

U,D,VT = np.linalg.svd(A)   #V is already transposed

print(U,D,VT,sep="\n")

# make D with right dimensions for multiplications
D = np.diag(D)  #Shape(2,2)
D = np.concatenate((D,[[0,0]]),axis = 0)   #Shape(3,2)
print(np.dot(U,np.dot(D,VT)))   #Same as original A

# Ex 2
A = np.array([
    [1,3,2],
    [4,6,8],
    [7,5,9]
])

S,D,VT = np.linalg.svd(A)
D = np.diag(D)
print(np.dot(S,np.dot(D,VT)))

# Ex3
A = np.array([
    [3,5,2],
    [8,6,7]
])

S,D,VT = np.linalg.svd(A)
D = np.diag(D)
D = np.concatenate((D,[[0],[0]]),axis=1)
print(np.dot(S,np.dot(D,VT)))