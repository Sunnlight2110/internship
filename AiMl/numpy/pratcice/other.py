import numpy as np
import matplotlib.pyplot as plt

# 5
X = np.random.randint(0,10,(4,4))
X1 = X.reshape(2,8)
print("reshape",X1)

print('flip horizontal',np.fliplr(X1))  #Reverse column order
print('flip Vertical',np.flipud(X1))    #Reverse row order

# 11
S,V,D = np.linalg.svd(X)
print('S',S)
print('V',V)
print('D',D)

# 12
X = np.random.randint(0,20,(5,5))
Xinv = np.linalg.inv(X)
print("inverse",Xinv)
print("Xinv.X",np.dot(X,Xinv).round())

# 13
X = np.random.randint(0,100,(2,6)) #dataset
print("polynomial",np.polyfit(x = X[0,:],y=X[1,:],deg=3))   

# 14
x = np.random.normal(0,1,1000)
y = np.random.normal(0,1,1000)
x_position = np.cumsum(x)
y_position = np.cumsum(y)
plt.plot(x_position, y_position, color='blue', alpha=0.7, label='Random Walk Path')
plt.scatter(x_position[-1], y_position[-1], color='red', label='Final Position')
plt.title('random walk')
plt.show()
