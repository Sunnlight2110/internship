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


A = np.array([
    [-1,4],
    [2,-2]
])

lambdas,V = np.linalg.eig(A)    
#Matrix V contain as many eigenvectors as columns of A, lambdas contain each eigenvalues fo crossponding eigenvectors in V
print(lambdas)
print(V)
v = V[:,0]
Av = np.dot(A,v)

plot_vectors([Av,v],['blue','lightblue'])
plt.xlim(-1,2)
plt.ylim(-1,2)
plt.show()

