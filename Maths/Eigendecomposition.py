import numpy as np

"""EQ 
    A = V(diag(lembdas))(V^-1)"""

# A = np.array([
#     [4,2],
#     [-5,-3]
# ])

# lambdas,V = np.linalg.eig(A)    #Eigen values

# V_inv = np.linalg.inv(V)
# lambdas = np.diag(lambdas)

# print(np.dot(V,np.dot(lambdas,V_inv)))  #Note ans = A

"""Eigendecomposition is not possible with all matrices. And in some cases where it is possible, the eigendecomposition involves complex numbers instead of straightforward real numbers.

In machine learning, however, we are typically working with real symmetric matrices, which can be conveniently and efficiently decomposed into real-only eigenvectors and real-only eigenvalues. If A is a real symmetric matrix then...

Α = QΛQT

...where Q is analogous to V from the previous equation except that it's special because it's an orthogonal matrix."""

A = np.array([      #A = real symmetric matrix
    [2,1],
    [1,2]
])

lambdas,Q = np.linalg.eig(A)    #Q = Orthogonal 
lambdas = np.diag(lambdas)
print(np.dot(Q,np.dot(lambdas,Q.transpose())))


            
"""
2D geometric transformation      2x2 Matrix         Eigenvalues             Example eigenvectors
Scaling (equal)                [[k, 0], [0, k]]     X₁ = X = k 2            non-zero
Scaling (unequal)              [[k₁, 0], [0, k₂]]   X₁ = k₁ and X₂ = K2     v₁ = [1,0] and v₂ = [0,1] 1
Horizontal shear               [[1, k], [0, 1]]     X₁ = X = 1 2            v₁ = [1,0] 1
Vertical shear                 [[1, 0], [k, 1]]     X₁ = X = 2 2            v1 = [0,1] 1
"""

