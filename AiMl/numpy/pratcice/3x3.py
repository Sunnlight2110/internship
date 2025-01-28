import numpy as np

# 1
X = np.random.randint(0,20,(3,3))
print('sum',np.sum(X))
print('mean',np.mean(X))
print('std',np.std(X))

# 8
X1 = np.random.randint(0,10,(2,3))
X2 = np.random.randint(0,10,(3,2))
print('multiplication',np.dot(X1,X2))

# 9
x = np.random.randint(1,10,3)
print('broadcasting',X1+x)

# 10
X = np.random.randint(0,20,(3,3))
e_value,e_vector = np.linalg.eig(X)
print("eigenvalue",e_value)
print("eigenvector",e_vector)
