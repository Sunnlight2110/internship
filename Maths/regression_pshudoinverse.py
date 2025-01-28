import numpy as np
import matplotlib.pyplot as plt
"""• Would be unusual to have exactly as many cases (n) as features (m)
 With pseudoinverse X+, we can now estimate model weights w if n ≠ m: w = X+y
 If X is overdetermined (n > m), X+ provides Xy as close to w as possible (in terms of Euclidean distance, specifically ||Xy - w||2)
 If X is underdetermined (n < m), X+ provides the w = Xty solution that + has the smallest Euclidean norm ||*||, from all the possible solutions 2"""

# We have 8 data points
x1 = [0,1,2,3,4,5,6,7]  #e.g. Dosage of drug to treating diseases
y = [1.86,1.31,0.62,0.33,0.09,-.67,-1.23,-1.37] #e.g. Effect of drug


# ====================================== Plotting
fig,ax = plt.subplots()
# plt.title('Clinical trial')
# plt.xlabel('Dosage')
# plt.ylabel('Effects')
# _= ax.scatter(x1,y)

"""Although it appears there is only one predictor (x1), 
     we need a second one (let's call it x0) in order to allow for a y-intercept (therefore, m = 2).
 Without this second variable, the line we fit to the plot would need to pass through the origin (0, 0).
 The y- intercept is constant across all the points so we can set it equal to 1 across the board:"""

# ? added X0 because with x1 we do not have y intercept(b), x0 adds y intercept,
# ? and reason for once is that it does not change volume(w) of original x1
# ? If x1 has b intercept, we will not need x0
x0 = np.ones(8)

    
X = np.concatenate((np.matrix(x0).T,np.matrix(x1).T),axis=1)
# print(X)

w = np.dot(np.linalg.pinv(X),y)     #because of w = Xplus * y
print(w)    #returns[y-intercept, slope of line]

b = np.asarray(w).reshape(-1)[0]    #make scatter of y-intercept
m = np.asarray(w).reshape(-1)[1]    #Make scatter of slope

plt.title('Clinical trial')
plt.xlabel('Dosage')
plt.ylabel('Effects')
_= ax.scatter(x1,y)

x_min, x_max = ax.get_xlim()
y_min, y_max = m*x_min + b, m*x_max + b     #Regression line: eq: y = mx + b where m = slope, b = y intersept(bias)

ax.set_xlim([x_min,x_max])
_= ax.plot([x_min,x_max],[y_min,y_max])
plt.show()
