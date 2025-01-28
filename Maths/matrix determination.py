import numpy as np
import matplotlib.pyplot as plt
def plot_vectors(vectors:list,color:list)->None:
    """
    Plot ine or more vectors in to 2D plane, Specifing colour for each
    
    args:   vectors: list full of numpy vectors
            colors: List full of colors for vectors
    """

    plt.figure()
    plt.axvline(x=0,color = "gray")
    plt.axhline(y=0,color = "gray")

    for i in range(len(vectors)):
        x = np.concatenate([[0,0],vectors[i]])
        plt.quiver([x[0]],[x[1]],[x[2]],[x[3]],angles = 'xy',scale_units = 'xy',scale=1,color = color[i])   #Gives arrows to vector for its direction
def vectorify(matrix,column):
    """Function to convert a column of matrix into vector"""
    return np.array(matrix[:,column]).reshape(-1)

"""Maps square matrix to scaler
    if det = 0,
        matrix can not be incerted
"""

X = np.array([
    [4,2],
    [-5,-3]
])
 
print(np.linalg.det(X))

X = np.array([
    [-4,1],
    [-8,2]
])
print(np.linalg.det(X))

X = np.array([
    [1,2,4],
    [2,-1,3],
    [0,5,1]
])
print(np.linalg.det(X))

"""det(X) = product of all eigenvalues of X

|det(X)| quantifies volume change as a result of applying X:
    If det(X) = 0, then X collapses space completely in at least one dimension, thereby eliminating all volume
    If 0 < |det(X)| < 1, then X contracts volume to some extent
    If |det(X)| = 1, then X preserves volume exactly
    If |det(X)| > 1, then X expands volume
"""

B = np.array([
    [1,0],
    [0,1]
])

N = np.array([
    [-4,1],
    [-8,2]
])

print(np.linalg.det(N)) #Zero, so it will collaps compleatly in atleast 1 dimension

NB = np.dot(N,B)    #B alone made a square, but after myltipling, B does not makes any shape

# plot_vectors([vectorify(B,0),vectorify(B,1),vectorify(N,0),vectorify(N,1)],
#              ['lightblue','lightgreen','blue','green'])

# plt.xlim(-6,6)
# plt.ylim(-9,6)

J = np.array([
    [-0.5,0],
    [0,2]
])
np.abs(np.linalg.det(J))    #1 means, This will not change the volume of matrix, rather just direction and volume of vectors
JB = np.dot(J,B)

plot_vectors([vectorify(B,0),vectorify(B,1),vectorify(JB,0),vectorify(JB,1)],
            ['lightblue','lightgreen','blue','green'])

plt.xlim(-1,3)
plt.ylim(-1,3)

plt.show()


