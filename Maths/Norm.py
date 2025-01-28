import numpy as np
point = np.array([25,2,5]) 

# * sum gives sum of all in array, abs gives absolute number of all array

# ? l1 norm
print('l1 norm ',np.sum(np.abs(point)))     #Manhattan distance

# ? L2 norm
print("euclidian distance = ", np.linalg.norm(point))  #((25^2)+(2^2)+(5^2))^0.5

# ? squared l2 norm
print("squared norm ",np.sum(point**2))

# ? Max norm
print("maxed ",np.max(np.abs(point)))

# ? generalized
p = 3
print('generalized ',np.sum(np.abs((point**p)**(1/p)))) # p1 = l1 norm, p2 = l2 norm, p infinity = max norm

# ? Frobenius norm (For matrix not vectors)
np1 = np.array([[1,2],[3,4]])
print('frobenius ',np.linalg.norm(np1,'fro'))

