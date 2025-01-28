import numpy as np

"""Trace operator
Tr(A) = sum of all diagonal elements
"""
A = np.array([
    [25,2],
    [5,4]
])

TrA = np.trace(A)
print(TrA)      #In this case diagonal = 25+4 = 29