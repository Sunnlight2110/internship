import numpy as np

# 1D
print("1D array")
np1 = np.array([i for i in range(1,11)])

for i in np1:
    print(i)

# 2D array
print("2D array")

np2 = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12]])
for i in np2:
    print(i) #prints rows
    for j in i:
        print(j)#prints data

# 3D array
np3 = np.array([i for i in range(0,12)])
np3 = np3.reshape(2,2,3)
for i in np3:
    for j in i:
        print(j)#prints rows
        for k in j:
            print(k)#prints data


# np.nditer()
print("nd.nditer()")
for i in np.nditer(np3):
    print(i)