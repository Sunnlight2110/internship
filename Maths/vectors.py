import numpy as np

np1 = np.array([12,32,22])
print(np1,np1.shape)
np1_t = np.transpose(np1)
print(np1_t,np1_t.shape)    #No effect for vector because of only 1 specified dimensions

np1 = np.array([[12,22,32]])
print(np1,np1.shape)
np1_t = np.transpose(np1)
print(np1_t,np1_t.shape)    #Transpose because 2 specified dimensions