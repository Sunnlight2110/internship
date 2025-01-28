import numpy as np
from random import choice

np1 = np.array([3,4,5,2,6,1,7,8,9])
print("unsorted",np1)
print("sorted",np.sort(np1))

# alphabetical
np2 = ["Himanshu","Tejashree","Divyesh","Shobhaben","Mahendrabhai"]
print("unsorted",np2)
print("sorted",np.sort(np2))

# bool
np3 = np.array([choice([True,False]) for i in range(0,5)])
print("unsorted",np3)
print("sorted",np.sort(np3))

# sort does not change original
print("unsorted",np2)
print("sorted",np.sort(np2))
print("unchanged",np2)

# 2D array
np4 = np.array([[3,4,2,5,6],[1,7,9,8,0]])
print("unsorted",np4)
print("sorted", np.sort(np4))