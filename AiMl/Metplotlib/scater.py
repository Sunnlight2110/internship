import matplotlib.pyplot as plt
import numpy as np

x = np.random.rand(50) #50 random values between 0 and 1
y = 2 * x + np.random.randn(50) * 0.1   #liner relation with noise


plt.scatter(x,y,color = 'blue', label = 'lmao') #Scatter plot for x and y

plt.xlabel('x axis')
plt.ylabel('y axis')
plt.grid(True)
plt.legend()
plt.show()