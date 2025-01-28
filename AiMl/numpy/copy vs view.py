import numpy as np
np1 = np.array([0,1,2,3,4,5])

# view
print("view")
np2 = np1.view()

print(f"np1 {np1}, np2 {np2}")

np1[0] = 10
print(f"changing a number in np1 \n np1 {np1}, np2 {np2}")


# copy
np1 = np.array([0,1,2,3,4,5])
print("copy")
np2 = np1.copy()

print(f"np1 {np1}, np2 {np2}")

np1[0] = 10
print(f"changing a number in np1 \n np1 {np1}, np2 {np2}")