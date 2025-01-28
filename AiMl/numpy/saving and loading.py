import numpy as np
"""
.npy/.npz: Best for working within NumPy, highly efficient and fast.
Text files (.txt, .csv): For compatibility with external tools or manual inspection.
Pickle: For saving Python objects or mixed data types.
Memory Mapping: For large arrays that cannot fit in memory.
"""

"""1. .npy formate(Native binary formate for numpy)"""

# np1 = np.arange(1,100)
# np.save('lmao.npy',np1)
# print("np1 saved")

# np1 = np.load('lmao.npy')
# print("array loaded from file, ",np1)


"""2. .npz(save multiple arrays in single file)"""

# np1 = np.arange(1,101)
# np2 = np.arange(200,401)

# np.savez("multi.npz", np1 = np1, np2 = np2)

# loaded = np.load('multi.npz')
# print(f"loaded np1 {loaded['np1']}, loaded np2 {loaded['np2']}")

"""3. saving as text file"""
# np1 = np.arange(1,101)
# np.savetxt('lol.txt',np1, delimiter=',', fmt="%d")

# loaded = np.loadtxt('lol.txt', delimiter=',')
# print("loaded ", loaded)

"""4. Using pickle for saving"""
# import pickle
# np1 = np.arange(1,101)
# with open('pickle.pkl','wb') as f:
#     pickle.dump(np1,f)
# with open('pickle.pkl','rb') as f:
#     loaded = pickle.load(f)
#     print("loaded ",loaded)

"""5. Advanced techniques"""
# compress nyz to save storage

# np1 = np.arange(1,101)
# np.savez_compressed('gg.nyz',np1 = np1)
# loaded = np.load('gg.nyz.npz')
# print('loaded ',loaded['np1'])

# For working with large arrays, use memory mapping
np1 = np.memmap('ff.dat', dtype=np.float16, mode='w+', shape=(1000,1000))

np1[:] = np.random.random((1000,1000))

subset = np1[100:200]

del np1
print("subset ",subset)
