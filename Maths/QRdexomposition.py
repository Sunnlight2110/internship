import numpy as np

"""QR decomposition is a way to factorize a matrix A into two simpler matrice

A=QR 
    Where:
        A is the original matrix (which can be rectangular)
        Q is an orthogonal matrix.
        R is an upper triangular matrix.    (Upper triangular means lower non diagonal are zero)"""

"""
Uses:

Solving linear systems
    Least squares regression
    Eigenvalue calculations
    Numerical stability
    Matrix factorization

Assumptions:
    Full column rank of A
    Works for both square and rectangular matrices Q has orthonormal columns
    Higher computational cost compared to LU in some cases"""

# ex 1
A = np.array([
    [1,2],
    [3,4],
    [5,6]
])

Q,R = np.linalg.qr(A)

print(Q,R,sep='\n')

print('reconstructed',np.dot(Q,R))

