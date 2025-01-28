import numpy as np
""" Two-dimensional array of numbers

    Denoted in uppercase, italics, bold, e.g.: X

    Height given priority ahead of width in notation, i.e.: (n.row' nco) col-
        If X has three rows and two columns, its shape is (3, 2)

    Individual scalar elements denoted in uppercase, italics only
        Element in top-right corner of matrix X above would be X[1.2 ]
    Colon represents an entire row or column:
        Left column of matrix X is X [:,1]
        Middle row of matrix X is X [2,:]"""

X = np.array([[1,2],[3,4],[5,6]])
print(X[0,1])
print(X[2,:])
print(X[0:2,0:1])

# ! Transposition
print(X)
print(X.transpose())
