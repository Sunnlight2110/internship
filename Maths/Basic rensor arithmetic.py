import numpy as np

"""Adding or multiply with scaler does not change tensor shape"""
print("scaler")
print(np.array([[1,2],[3,4]]) *2)

"""If tensors are same size, opretion default applied element wise,
called hadamard or element-wise product"""
print("same size tensors")
print(np.array([[1,2],[3,4]]) + np.array([[5,6],[7,8]]))

"""Reduction
Calculate sum of all elements in tensor"""
print("Reduction")
print(np.sum(np.array([[1,2,3],[4,5,6],[7,8,9]])))

"""same can be done with maximum,minimum,mean and product"""
np1 = np.random.randint(1,10,(2,2))
print("Max",np.max(np1))
print("Min",np.min(np1))
print("Mean",np.mean(np1))
print("Product",np.prod(np1))  #Multiplication of all elements and then sum of all

"""Dot product
Sum of product of all elements in tensors"""
np2 = np.random.randint(1,10,(2,2))
print("Dot product",np.dot(np1,np2))

"""Matrix multiplications
number of rows in first matrix = number of cols in second matrix
    i.g. c(i,k) = a(i,j) b(j,k)"""

print('multiply ',np.multiply(np1,np2))
