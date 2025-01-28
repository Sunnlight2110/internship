import scipy
import numpy as np
import scipy.linalg

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