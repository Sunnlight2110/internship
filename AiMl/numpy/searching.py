import numpy as np

np1 = np.array([1,2,3,4,5,6,7,8,9,3])

# where
x = np.where(np1 == 3)
print(np1)
print("where np1 == 3",x)
print(np1[x])
y = np.where(np1 % 2 == 0)
print("even items", y, np1[y])
