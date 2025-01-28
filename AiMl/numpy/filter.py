import numpy as np
from random import choice

np1 = np.array([1,2,3,4,5,6,7,8,9,10])
x = [choice([True,False]) for i in range(1,11)]

print(np1)
print("filtered",np1[x])

x = []
for i in np1:
    if i % 2 == 0:
        x.append(True)
    else:
        x.append(False)

print(np1)
print("filtered", np1[x])

# shortcut
x = []
x = np1 % 2 == 0
print(np1)
print("filtered", np1[x])