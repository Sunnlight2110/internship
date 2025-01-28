import numpy as np

# numpy = numeric python
# ndarray = n-diamentional array

np1 = np.array([i for i in range (0,10)])

print(np1.shape)

np2 = np.arange(10)
print(np2)

np3 = np.zeros(10)
print(np3)

np4 = np.zeros((2,10))
print(np4)

np5 = np.full((10),6)
print(np5)

# multidimensional array
print("multidimensional array")
np7 = np.full((2,10),6)
print(np7)

# access element
print("same as the list")
print(f"access the nth element{np1[2]}")
print(f"slicing {np1[3:8]}")


print()
np8 = np.array([[i for i in range(0,10)],[i for i in range(10,20)]])
print("2d array")
print(f"get nth element {np8[1,2]}")
print(f"slicing {np8[0:1,2:9]}")
print(f"slicing {np8[0:3,2:9]}")
print(f"slicing {np8[1,2:9]}")