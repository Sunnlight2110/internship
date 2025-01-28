import numpy as np
import matplotlib.pyplot as plt

# Number of degrees of freedom (df)
df = 2

# Generate 1000 random values from a Chi-Square distribution with df=2
data = np.random.pareto(a=0.99, size=(2, 3))
print(data)

# Plot the distribution
# plt.hist(data, bins=30, density=True, alpha=0.6, color='g')

# Add a title and labels
# plt.title("Chi-Square Distribution (df=2)")
# plt.xlabel("Value")
# plt.ylabel("Density")

# plt.show()
