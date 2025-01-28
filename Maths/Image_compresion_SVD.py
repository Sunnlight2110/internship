from PIL import Image   #PIL = pillow library for image processing
import numpy as np
import matplotlib.pyplot as plt

img = Image.open('dead-space.jpg')
gray = img.convert('LA')

#? =============================== Convert image data into numpy matrix
imgmat = np.array(list(gray.getdata(band=0)),float) #Creates one dimension array(vector)
imgmat.shape = (gray.size[1],gray.size[0])
imgmat = np.matrix(imgmat)  #Converts into matrix

U,sigma,V = np.linalg.svd(imgmat)

""" A = UDV
As eigenvalues are arranged in descending order in diag(lembda) so to are singular values, by convention, 
arranged in descending order in D (or, in this code, diag(sigma)).
Thus, the first left-singular vector of U and first right-singular vector of V may represent the most prominent feature of the image"""

# # ? reconstruct using first left-singular value of U and first right singular value of V, first singular value of sigma
# reconstruct = np.matrix(U[:,:1]) * np.diag(sigma[:1]) * np.matrix(V[:1,:])

# # ? reconstruct using i-th left-singular value of U and i-th right singular value of V, i-th singular value of sigma
# for i in [2**i for i in range(1,7)]:
#     reconstruct = np.matrix(U[:,:i]) * np.diag(sigma[:i]) * np.matrix(V[:i    ,:])


i = 50
reconstruct = np.matrix(U[:,:i]) * np.diag(sigma[:i]) * np.matrix(V[:i,:])

print(imgmat.shape)
full_representation = 1080*1920
compresed_representation = (i*1080)+i+(i*1920)
print("full representation = ",full_representation)
print("comparsed representation = ",compresed_representation)
_=plt.imshow(reconstruct,cmap='gray')

plt.show()
