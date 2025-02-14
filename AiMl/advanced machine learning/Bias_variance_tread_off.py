import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics

curve_df = pd.read_csv('C:\\Users\\My PC\\Desktop\\Internship\\Machine Learning (Codes and Data Files)\\Data\\curve.csv')
# print(curve_df.head(10))

# plt.scatter(curve_df.x,curve_df.y)
# plt.xlabel('x values')
# plt.ylabel('y values')
# plt.show()

def fit_poly(degree):
    """
    Fits a polynomial of a given degree to the data in curve_df and plots the original data along with the fitted curve.
    Parameters:
    degree (int): The degree of the polynomial to fit.
    Returns:
    list: A list of Line2D objects representing the plotted fitted curve.
    Notes:
    - This function uses the global variable `curve_df` which should be a DataFrame containing 'x' and 'y' columns for the data points.
    - The function also uses `np.polyfit` to fit the polynomial and `sn.regplot` to plot the original data points.
    - The fitted curve is plotted using `plt.plot`.
    """
    # ployfit gets best plynomial curve to minimum errors(distance)
    p = np.polyfit(curve_df.x,curve_df.y,deg=degree)    #degree 1 = straight line, degree 2 = parabola
    # calculates the predicted y values for the given x using the polynomial equation. y = a x^n + b x^n-1 + c x^n-3 + ... + dx + e
    curve_df['fit'] = np.polyval(p,curve_df.x)      #Calculate y points using equations
    sns.regplot(x=curve_df.x, y=curve_df.y, fit_reg=False)  #Create scatter plot, fit_reg = false: dont add regression line, cause we are adding our own
    return plt.plot(curve_df.x,curve_df.fit,label = 'fit')

# fit_poly(1)   #under fitting model has large error because of hight bias
# fit_poly(2)   #Find somewhere in between
# fit_poly(10)  #Over fitting model has large error because of high variance

plt.xlabel('x values')
plt.ylabel('y values')
# plt.show()

train_x,test_x,train_y,test_y = train_test_split(
    curve_df.x,
    curve_df.y,
    test_size=0.4,
    random_state=100
)

rmse_df = pd.DataFrame(columns=['degree','rmse_train','rmse_test'])

for i in range(1,15):
    p = np.polyfit(train_x,train_y,deg=i)
    rmse_df.loc[i-1] = [
        i,
        np.sqrt(metrics.mean_squared_error(train_y,np.polyval(p,train_x))),
        np.sqrt(metrics.mean_squared_error(test_y,np.polyval(p,test_x))),
    ]

# print(rmse_df)

plt.plot(
    rmse_df.degree,
    rmse_df.rmse_train,
    label = 'Rsme on training set',
    color = 'r'
)
plt.plot(
    rmse_df.degree,
    rmse_df.rmse_test,
    label = 'Rsme on Testing set',
    color = 'g'
)

plt.legend(
    bbox_to_anchor = (1.05,1),  #moves legends outside of plots
    loc = 2,                    #places legends upper left corner
    borderaxespad = 0           #No extra spacing
)
plt.xlabel('Model degrees')
plt.ylabel('RMSE')
plt.show()

"""
Error for train is hight at degree 1 and 15,
Error on test reduces initially, but gets up after a specific point
"""