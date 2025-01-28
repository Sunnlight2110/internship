import numpy as np

np1 = np.array([i for i in range(-5,10)])

# square root of each element
print(f"np.sqrt(np1) {np.sqrt(np1)}")

# absoulute value(distance from zero)
print(f"np.absolute(np1){np.absolute(np1)}")

# exponants(e^x, e = 2.718)
print(f"np.exp(np1){np.exp(np1)}")

# min/max
print(f"np.min(np1) {np.min(np1)}, np.max(np1) {np.max(np1)}")

# sign negetive or positive
print(f"np.sign(np1){np.sign(np1)}")

# trig(trigonomatory)(sin, cos, log)
print(f"np.sin(np1) {np.sin(np1)},\n np.cos(np1) {np.cos(np1)}, \n np.log(np1) {np.log(np1)}")


# https://numpy.org/doc/stable/reference/ufuncs.html  numpy universal functions