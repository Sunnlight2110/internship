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
# country_df_cleaned = country_df.drop(index=10)  # Index starts from 0, so 11th point is at index 10


# split data
Y = country_df['Corruption_Index']
X = sm.add_constant(country_df['Gini_Index'])
train_x,test_x,train_y,test_y = train_test_split(X,Y, train_size=0.7, random_state=45)

country_model_1 = sm.OLS(train_y,train_x).fit()
print(country_model_1.summary2())
#in summery slope is -0.8224, means change in curroption unit in unite change in Gini index = -0.8224
#  p-value (0.2847) is greater than 0.05, we conclude that there is no statistically significant relationship between the Gini Index and the Corruption Index in this model.

# # check residuals
# stats.probplot(country_model_1.resid, dist="norm", plot=plt)
# plt.title('Probability Plot of Residuals')
# plt.show()

# # cooks distance
# influence = country_model_1.get_influence()
# cooks_d = influence.cooks_distance[0]
# plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt="o", basefmt=" ")
# plt.title('Cook\'s Distance for Each Data Point')
# plt.xlabel('Index')
# plt.ylabel('Cook\'s Distance')
# plt.show()

# # check leverage
# leverage = influence.hat_matrix_diag
# plt.stem(np.arange(len(leverage)), leverage, markerfmt="o", basefmt=" ")
# plt.title('Leverage for Each Data Point')
# plt.xlabel('Index')
# plt.ylabel('Leverage')
# plt.show()  # 11 Is high influential point

# # Plot the residuals to check homoscedasticity.
# plt.scatter(country_model_1.fittedvalues, country_model_1.resid)
# plt.axhline(0, color='red', linestyle='--')
# plt.title('Residuals vs Fitted')
# plt.xlabel('Fitted Values')
# plt.ylabel('Residuals')
# plt.show()

pred_y = country_model_1.predict(test_x)

# answer of second qauction
print(np.abs(r2_score(test_y,pred_y)))
print(np.sqrt(mean_squared_error(test_y,pred_y)))

# answer of last
conf_int_96 = country_model_1.conf_int(alpha=0.04)  #Its CI, alpha = 100-96 = 4%
print(conf_int_96)

