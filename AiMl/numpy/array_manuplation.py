import numpy as np
"""
Reshaping: reshape(), resize(), flatten(), ravel()
Slicing: Indexing and slicing with :
Stacking: Combine arrays with vstack(), hstack(), and dstack()
Splitting: Split arrays using split(), hsplit(), vsplit()
Modifying: append(), delete()
"""

# np1 = np.array([1,2,3,4,5,6,7,8,9])
# # Reshape
# np2 = np1.reshape(3,3)
# print("np1 reshaped as np2", np2)

# # Resize (reshape original array)

# np1.resize(3,3)
# print("np1 resized",np1)

# # Flattening (Convert multi dimensional arrays into 1D),ravel(similar to flatten but returns a view and affects the original)
# np3 = np1.flatten()
# print("np1 flattened as np3",np3)

# # Stacking arrays
# np1 = np.array([1,2,3])
# np2 = np.array([4,5,6])
# np3 = np.vstack((np1,np2)) #Stacks arrays vertically
# print("vstack np1 and np2 into np3",np3)

# np3 = np.hstack((np1,np2)) #Stacks arrays horizontally
# print("hstack np1 and np2 into np3",np3)

# # array splitting, hsplit
# np1 = np.array([1,2,3,4,5,6,7,8])
# np2 = np.split(np1,4)
# print("splitting np1 into np2",np2)

# np1 = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
# np2 = np.vsplit(np1,2)
# print("vsplit np1 into np2",np2)

# Adding and removing  into 1D array
np1 = np.array([1,2,3,4,5,6,7,8])
np2 = np.append(np1,[9,10])
print("Append into np1",np2)

np2 = np.insert(np1,2,[3.141]) #Add at specific index
print("Insert into np1",np2)

np2 = np.delete(np1,3) #Delete a specific index
np2 = np1[np1 != 3] #Removes all values specified
print("Delete from np1",np2)

# Adding and removing from 2D arrays
np1 = np.array([[1,2,3],[4,5,6]])
# axis 0 adds new row, axis 1 adds new columns
np2 = np.append(np1,[[7,8,9]],axis=0)
print("axis = 0",np2)
np2 = np.append(np1,[[7],[8]],axis=1)
print("axis = 1", np2)

np2 = np.insert(np1,1,[7,8,9],axis=0)
print("axis = 0",np2)
np2 = np.insert(np1,1,[7,8],axis=1)
print("axis = 1",np2)

np2 = np.delete(np1,1,axis=0)
print("axis = 0",np2)

np2 = np.delete(np1,1,axis=1)
print("axis = 1",np2)
