import scipy
import numpy as np
import scipy
import scipy.linalg

"""LU works if:

A is square.
A is non-singular (det(A)!=0).
No zero pivots (or we use pivoting).
If conditions fail, use:
Partial Pivoting (PA = LU) for numerical stability.
QR decomposition if 
A is not square."""

A = np.array([
    [4,1,-1],
    [3,6,-1],
    [2,1,5]
])

# Perform decomposition
P,L,U = scipy.linalg.lu(A)

# Print the matrices
print("P (Permutation Matrix):")
print(P)

print("\nL (Lower Triangular Matrix):")
print(L)

print("\nU (Upper Triangular Matrix):")
print(U)

# Verify by multiplying P, L, and U to get A back
print("\nReconstructed Matrix A (P * L * U):")
print(np.dot(P, np.dot(L, U)))


# ? Example
A = np.array([
    [3,2,-1],
    [2,3,2],
    [1,2,3]
])
B = np.array([4,5,6])

P,L,U = scipy.linalg.lu(A)

y = scipy.linalg.solve(L,B)
X = scipy.linalg.solve(U,y)
print(X)
