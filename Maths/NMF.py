import numpy as np
from sklearn.decomposition import NMF
"""
NMF is a powerful technique in linear algebra that factorizes a matrix into two smaller matrices, typically W and H, where both of these matrices contain only non-negative values (i.e., no negative numbers allowed!).
This is often used for dimensionality reduction or feature extraction, especially in the realms of machine learning and data mining.
"""

"""
V≈W*H
Where:

V is the original matrix (usually data),
W is a matrix of features (or basis vectors),
H is the matrix of coefficients (or activations)."""

V = np.array([[5, 3, 0, 1],
              [4, 0, 0, 1],
              [1, 1, 0, 2],
              [2, 4, 3, 5]])

# Initialize NMF model (set number of components)
"""n_components=2: This means you're asking NMF to break the matrix 
V into 2 components (i.e., 2 basis vectors/features). It's like you're saying, "I want to learn two features from my data."

init='random': You're initializing the matrices 
W and H randomly. These matrices will get adjusted during the factorization process to approximate the original matrix V.

random_state=42: This ensures the random initialization is the same every time you run the code (for reproducibility). You could use any number here, but 42 is a classic choice for randomization."""

nmf = NMF(n_components=2, init='random', random_state=42)

# Fit the model and get the factorized matrices W and H
"""nmf.fit_transform(V): This is where the magic happens! You're fitting the NMF model to the matrix 
V, which means it tries to learn the best possible approximation for V as the product of 
W and H. The fit_transform function not only fits the model but also returns the matrix 
W (the features matrix).

nmf.components_: After fitting, nmf.components_ gives you the matrix 
H (the coefficients matrix), which contains the activations of the features in 
W."""

W = nmf.fit_transform(V)
H = nmf.components_


print("Matrix W (Features):")
print(W)

print("\nMatrix H (Coefficients):")
print(H)

# Verify the decomposition by reconstructing V
reconstructed_V = np.dot(W, H)
print("\nReconstructed Matrix V (W * H):")
print(reconstructed_V)

