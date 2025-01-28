import numpy as np

"""Boolean Indexing: Select elements based on conditions.
Fancy Indexing (Integer Indexing): Use integer arrays to index elements, rows, or columns.
Slicing + Fancy Indexing: Use both slicing and arrays together to select elements.
Conditional Replacement: Replace elements based on conditions using Boolean indexing.
np.ix_(): Create index arrays that work like meshgrid to select rows and columns.
"""

# Boolean indexing(Using boolean condition to indexing array)
np1 = np.random.randint(1,20,10)

condition = np1 > 10
print("Boolean indexing for np1 > 10", np1[condition])

# 2D fancy indexing
np1 = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])

rows = [0,1]
result_2d = np1[rows]
print("2d fancy indexing",result_2d)

# Indexing with slicing in array
print("indexing with slicing",np1[2:3,1:])

# Conditional replacement with advanced indexing

np1[np1>7] = True
print("Conditional replacement",np1)

# using np.ix()(combine row and column indices into a meshgrid-like indexing)

np1 = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])
print("using np.ix()",np1[np.ix_([0,2],[1,4])])
