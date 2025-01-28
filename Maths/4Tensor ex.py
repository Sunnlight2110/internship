import numpy as np

"""Higher-Rank Tensors

As an example, rank 4 tensors are common for images, where each dimension corresponds to:

1. Number of images in training batch, e.g., 32

2. Image height in pixels, e.g., 28 for MNIST digits

3. Image width in pixels, e.g., 28

4. Number of color channels, e.g., 3 for full-color images (RGB)"""

images_pt = np.zeros([32,28,28,3])  #Creates zero 4tensor

