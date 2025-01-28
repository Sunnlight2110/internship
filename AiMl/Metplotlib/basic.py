import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,10,1000)
y = np.cos(x)
z = np.sin(x)

plt.plot(x,y,label = 'Sine wave') # adds plot to create line plot, label = adds label to the line
plt.plot(x,z,label = 'cos wave') # adds plot to create line plot, label = adds label to the line
plt.xlabel('x axis')    #adds x axis label
plt.ylabel('y axis')    #Adds y axis label

plt.title('Simple line plot') #Adds title to plot

plt.legend() #Display legend for label defined in plot

plt.grid(True)  #Adds grid to plot

plt.show()  #Displays plot