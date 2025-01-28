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

x = np.array([3,1])
color = ['lightgreen']

vector = [x]

# Applying matrix to vector can linearly transform vector(e.g. rotate or rescale) accept identity metrix

E = np.array([
    [1,0],
    [0,-1]
])

vector.append(np.dot(E,x)), color.append('green')

# Applying matrix to apply multiple transformation

A = np.array([
    [-1,4],
    [2,-2]
])
vector.append(np.dot(A,x)), color.append('darkgreen')
plot_vectors(vector,color)

x2 = np.array([2,1])

np.matrix(x).T  #Transpose matrix

x3 = np.array([-3,-1])
x4 = np.array([-1,1])

X = np.concatenate((
    np.matrix(x).T,
    np.matrix(x2).T,
    np.matrix(x3).T,
    np.matrix(x4).T,
),axis=1)

def vectorify(matrix,column):
    """Function to convert a column of matrix into vector"""
    return np.array(matrix[:,column]).reshape(-1)

vectorify(X,0)  #Returns x

AX = np.dot(A,X)
vector = [
    vectorify(X,0),
    vectorify(X,1),
    vectorify(X,2),
    vectorify(X,3),
    vectorify(AX,0),
    vectorify(AX,1),
    vectorify(AX,2),
    vectorify(AX,3),
]
color = [
    "lightblue",
    "lightgreen",
    'lightgray',
    'orange',
    'blue',
    'green',
    'gray',
    'red'
]

plot_vectors(vectors=vector,color=color)

plt.xlim(-5,5)
plt.ylim(-5,5)
plt.show()

