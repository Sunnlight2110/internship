import numpy as np

# Create 1D array
np1 = np.array([i for i in range(1,13)])
print(f"np1 {np1}")
print(f"np1.shape {np1.shape}")

# create 2D array returns (rows, columns)
np2 = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12]])
print(f"np2 {np2}")
print(f"np2.shape {np2.shape}")

# Reshape 2D
np3 = np1.reshape(3,4)
print(f"np1 reshape as np3 {np3}")
print(f"np3.shape {np3.shape}")

# Reshape 3D
np4 = np1.reshape(2,3, 2)
print(f"np1 reshape as np4 {np4}")
print(f"np4.shape {np4.shape}")

# Flatten to 1D
np5 = np2.reshape(-1)
print(f"np2 reshape as np5 {np5}")
print(f"np5.shape {np5.shape}")
