import numpy as np

# 2
x = np.random.randint(1,100,20)
print(x[:5])    #Print first 5 elements
print(x[3:8])   #Print index of 3-7

# 3
x = np.array([1,2,3,4,5,6,7,8,9])

y = x.reshape(3,3)     #Reshape into 3x3 matrix
print(y)
yT = y.T
print("Transpose",yT)    #Transpose

# 4
x = np.random.randint(0,100,30)
print("greater than 50",x[x>50])

# 6
print("max",np.max(x))
print('index',np.argmax(x))
print("min",np.min(x))
print('index',np.argmin(x))

# 7
x = np.random.randint(0,15,20)
print('unsorted',x)
x = np.sort(x)
print('sorted',x)
print('unique',np.unique(x))