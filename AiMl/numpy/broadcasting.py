import numpy as np

"""
    If the arrays have a different number of dimensions, 
        the smaller-dimensional array is padded with ones on the left side (until they have the same number of dimensions).

    The size of the dimensions must either be the same or one of them must be 1.
         If one of the dimensions is 1, the array is stretched (broadcast) to match the size of the other array along that dimension.
"""

"""
    steps:
        Compare the shapes of the two arrays element-wise, starting with the trailing dimensions (i.e., from the right side).
        If the dimensions are equal or one of them is 1, they are compatible for broadcasting.
        If the dimensions are different and neither is 1, broadcasting cannot be done, and an error is raised.
"""

scaler = 5

np1 = np.array([[1,2,3],[4,5,6]])

print(np1+scaler)

np2 = np.array([1,2,3])
print(np1+np2)# shape of np1(2,3), shape of np2(1,3) is similar

np3 = np.array([[1,2,3],[4,5,6]])

print(np.broadcast(np1+np3))