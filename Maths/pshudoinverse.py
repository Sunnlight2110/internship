import numpy as np
"""
SVD is only for squre matrix
Pshudoinverse is for non square matrix too

    A = V D U.T
    where V,D,U = SVD of A
    D = d with resiprocal(1,d) of all non zero 
"""

A = np.array([
    [-1,2],
    [3,-2],
    [5,7]
])

U,d,VT = np.linalg.svd(A)

# print(U,d,VT,sep="\n")

# ? to convert d in to resiprocal
D = np.diag(d)
Dinv = np.linalg.inv(D)   #Here D becomes D-plus

# ? make D same dimensions as AT for multiplications
Dplus = np.concatenate((Dinv,np.array([[0,0]]).T),axis=1)


# ? Calculate Pshudo inverse
Aplus = np.dot(VT.T,np.dot(Dplus,U.T))
print(Aplus)
print(np.linalg.pinv(Aplus))    #pinv = pshudo inverse, same as original array
