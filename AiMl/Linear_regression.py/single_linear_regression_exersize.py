import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import scipy.stats as stats

import matplotlib.pyplot as plt

"""
    country.csv contains corruption perception index and gini index of 20 countries
        index close to 100 means low corruption, close to 0 means high corruption

    1: develop simple linear regression model. what is change in corruption in every unite change in gini index.
    2: what proportion of variation in corruption index is explained by gini index
    3: is there a statistically significant relationship between corruption index and gini index?
    4: calculate 96% confidence for regression coefficient
"""

country_df = pd.read_csv('C:\\Users\\My PC\\Desktop\\Internship\\Machine Learning (Codes and Data Files)\\Data\\country.csv')

# split data
Y = country_df['Corruption_Index']
X = sm.add_constant(country_df['Gini_Index'])
train_x,test_x,train_y,test_y = train_test_split(X,Y, train_size=0.7, random_state=45)

country_model_1 = sm.OLS(train_y,train_x).fit()
print(country_model_1.summary2())
# check residuals
stats.probplot(country_model_1.resid, dist="norm", plot=plt)
plt.title('Probability Plot of Residuals')
# plt.show()

# cooks distance
influance = country_model_1.get_influence()
distance = influance.cooks_distance[0]
plt.stem()

