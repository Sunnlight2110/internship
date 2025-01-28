import numpy as np
"""Key concepts
Vectors: An array of numbers that can be treated as a single entity (e.g., [2, 4, 6]).
Matrices: 2D arrays of numbers (e.g., [[1, 2], [3, 4]]).
Dot Product: The sum of the product of corresponding elements of two vectors or matrices.
Matrix Multiplication: The process of multiplying two matrices.
Transpose: Flipping a matrix over its diagonal, turning rows into columns.
Inverse: The matrix that, when multiplied with the original matrix, yields the identity matrix.
Determinant: A scalar value that is computed from the elements of a square matrix.
Eigenvalues and Eigenvectors: Scalar values and corresponding vectors that are fundamental to matrix diagonalization.
"""

"""np.array(): Create a matrix or vector.
np.dot(): Dot product or matrix multiplication.
np.cross(): Cross product of two vectors.
np.transpose(): Transpose of a matrix.
np.linalg.inv(): Inverse of a matrix.
np.linalg.det(): Determinant of a matrix.
np.linalg.eig(): Eigenvalues and eigenvectors of a matrix.
np.linalg.solve(): Solve linear systems.
"""

# Dot product
np1 = np.array([1,2,3])
np2 = np.array([4,5,6])
print("dot",np.dot(np1,np2)) #(Sum of corresponding  elements)

# Cross product
print("Cross product ",np.cross(np1,np2)) #A vector that is perpendicular to plane

"""Matrix and matrix operations"""
# Matrix transpose
np1 = np.array([[1,2],[4,5]])
print("transpose ",np.transpose(np1))  #Converts rows into columns and columns into rows

# Matrix multiplications
np2 = np.array([[7,8],[10,11]])
print("Multiply matrix",np.dot(np1,np2)) #Multiply matrix (number of rows must be equal)
print("Element wise multiplication",np1*np2) #Element wise multiplication

# Inverse matrix
"""
The inverse of a square matrix A is the matrix that, when multiplied by A,
gives the identity matrix I. Not all matrices have inverses, must be square and non-singular (determinant not equal to zero).
"""
print("inverse",np.linalg.inv(np1))  #Returns inverse of matrix
print("determinant",np.linalg.det(np1)) #Show if matrix is invariable or not

# Eigenvalues and Eigenvectors
"""
Eigenvalues and eigenvectors are fundamental in matrix diagonalization, solving differential equations, 
and in machine learning algorithms like Principal Component Analysis (PCA).
"""

eiganvalues,eiganvectors = np.linalg.eig(np1)
print("eiganvalue",eiganvalues)
print("eiganvector",eiganvectors)

# Solving liner algebra(Ax = b)
A = np.array([[1,2],[3,4]])
b = np.array([11,12])

print("Solving",np.linalg.solve(A,b))


