import numpy as np
"""
y = Xw
where   y = outcome(price)
        X = features(bedroom count)
        w = unknown variables(model's learnable parameters)

y(X^-1) = (X^-1)Xw
y(X^-1) = w
"""

np1 = np.array([[4,7],[2,6]])
np2 = np.linalg.inv(np1)


print(np.round(np.dot(np2,np1)))

"""
        Only be calculated if :
                Matrix in not singular
                must be linearly independent
                        i.g. lines should touch only at a point
        
        Avoid over dertermination:
                num of rows > num. of cols
                        i.e. no. of eq > no. of dimensions
        Avoid over determination:
                no. of rows < no. of cols
                        i.e. no. of eq < no. of dimensions
        
"""