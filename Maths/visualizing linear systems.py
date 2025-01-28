import numpy as np
import matplotlib.pyplot as plt

"""eqs
    y = 3x, -5x+2y = 2
"""

x = np.linspace(-10,10,1000)

y1 = -5+(2*x)/3
y2 = (7-2*x)/5

fig,ax = plt.subplots()

ax.plot(x,y1,color='red')
ax.plot(x,y2,color='blue')

ax.set_xlim([0,10])
ax.set_ylim([-6,2.5])

# plt.axvline(x = 2, color = 'purple', linestyle = '--')
# plt.axhline(y = 6, color = 'purple', linestyle = '--')

plt.grid(True)
plt.show()
