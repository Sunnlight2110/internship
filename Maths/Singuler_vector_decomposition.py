import numpy as np

"""
As on slides, SVD of matrix A is:

A = UDVT

Where:

U is an orthogonal m X m matrix; its columns are the left-singular vectors of A. crossponds rows of output.
V is an orthogonal n x n matrix; its columns are the right-singular vectors of A. crossponds cols of output.
D is a diagonal m X n matrix; elements along its diagonal are the singular values of A. crossponds direction of output.
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
