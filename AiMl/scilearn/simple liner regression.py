import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import random

"""data example of years of experience and salary"""

x = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]) #Years of experience
y = np.array([15000,18000,21000, 24000, 27000, 30000, 33000, 36000, 39000, 42000])  #Salary

z = np.array([[random.randint(1,10)+random.random()]for i in range(0,10)])
print(np.shape(x))
print(np.shape(y))

model = LinearRegression()


model.fit(x,y)  #train model with data
predicted_salary = model.predict([[6.5]])  #Predict salary of 6.5 years of experience
print(f"predicted salary for 6.5 years of experience is {predicted_salary[0]:.2f}")

plt.plot(x,y,color='red',label='salary-experience')
plt.scatter(x,model.predict(z),color='blue')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.title('linear regression: salary vs Years of experience')
plt.show()