import numpy as np

"""Orthogonalization is a process in linear algebra where we transform a set of vectors into orthogonal vectors (vectors that are perpendicular to each other). If the vectors are normalized (have a length of 1), they become orthonormal.

This concept is super useful in machine learning, especially for creating independent features or basis vectors for spaces."""

# define array
v1 = np.array([1,1])
v2 = np.array([1,0])

# ? Normalization   (Changes magnitude of vectors, but do not guaranty to be vectors)
v1_normalized = v1/np.linalg.norm(v1)
v2_normalized = v2/np.linalg.norm(v2)

# ? check for orthogonal or not (Orthogonal are at 90 degree to each other, so dot = 0)
print(np.dot(v1_normalized,v2_normalized))

# ! Gram schmidt process
vectors = np.array([v1,v2])
orthogonalized = []
for v in vectors:
    ortho_v = v.astype(float)
    for u in orthogonalized:
        proj = np.dot(u,v)/np.dot(u,u) * u
        ortho_v -= proj
    orthogonalized.append(ortho_v)

orthogonalized_vectors = np.array(orthogonalized)

# ? normalize orthogonalized

orthonormal_vectors = orthogonalized_vectors/np.linalg.norm(orthogonalized_vectors,axis=1)[:,np.newaxis]

# ? check for orthogonal
print(np.dot(orthonormal_vectors[0],orthonormal_vectors[1]))
    