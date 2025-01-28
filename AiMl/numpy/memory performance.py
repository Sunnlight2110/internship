import numpy as np

"""Key Factors Affecting Memory Performance in NumPy
Reduce Data Type Precision: Use smaller dtype values (float32 instead of float64).
Use Views Instead of Copies: Avoid unnecessary duplication.
Leverage In-Place Operations: Use operations like +=, -=, etc.
Preallocate Memory: Use np.empty() or np.zeros() to allocate space before adding data.
Use Broadcasting Effectively: Avoid creating intermediate arrays.
Utilize Memory Mapping for Large Data: Use np.memmap for datasets too large for memory.
Monitor Memory Usage: Use libraries like memory_profiler to identify memory bottlenecks.
"""

"""
1. Data type 
The data type (dtype) of a NumPy array determines how much memory each element takes. 
By default, NumPy often uses 64-bit floats or integers, which might not always be necessary.
"""
np1 = np.arange(1,21)
print(f"Array {np1}, datatype {np1.dtype}")

np1 = np.arange(1,21,dtype=np.int16)
print(f"Array {np1}, datatype {np1.dtype}")

"""
2. Views and copy
View shares data with original while copy duplicates data into memory.
View saves memory
"""

"""
3. Inplace operations
Inplace operations modify array directly without creating new one.
Saves memory, reduces computation overhead
"""

np1 = np.array([1,2])

np2 = np1 + [21,22] # regular operations

np1 += [21,22] # inplace operations

"""
4. Efficient broadcasting
Avoids creating intermediat arrays
Saves memory and time
"""

np1 = np.array([1,2,3,4])

result = np1 * 2 #Broadcasting (No new array created)

result = np.array([2,2,2,2]) @ np1  #explicit ([2 2 2 2] created)

"""
5. Preallocating memory
Preallocating memory for large arrays is more efficient than dynamically appending to arrays.
Use np.empty() or np.zeros() to preallocate memory.
"""

# Dynamic growing array
np1 = np.array([])
for i in range(0,10):
    np.append(np1,i)

# preallocated memory
np1 = np.empty(10,dtype=np.int16)
for i in range(0,10):
    np.append(np1,i)

"""
6. Memory mapping
For extremely large arrays, use memory mapping (np.memmap) to avoid loading the entire array into memory.
"""

filename = 'lmao.dat'
large_array = np.memmap(filename=filename, dtype= np.float32, mode = 'w+', shape=(1000,1000))

large_array[:] = np.random.random((1000,1000))

subset = large_array[100:500]
print("subset = ",subset)


"""
7. Garbage collection and memory overhead
NumPy relies on Python's garbage collector to free memory. Explicitly deleting unused arrays can help
"""
# Create large array
large_array = np.random.random(100,100)

# free up memory by deleting array
del large_array

"""You can also use the gc module to force garbage collection"""
import gc
gc.collect()