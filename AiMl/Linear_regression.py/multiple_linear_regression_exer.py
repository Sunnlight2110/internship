import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt


"""This program performs Multiple Linear Regression on a given dataset to predict a target variable.
It utilizes multiple independent variables (predictors) to build a linear model and predict the dependent (target) variable.

The program works as follows:
1. The dataset should contain both independent variables (e.g., 'Square_Feet', 'Num_Rooms', 'House_Age') and the dependent variable (e.g., 'Price').
2. The model fits a linear equation to the data, estimating the relationship between the predictors and the target.
3. It calculates the coefficients for each independent variable, and uses these to make predictions.
4. Evaluation metrics like R-squared and Mean Squared Error (MSE) are used to assess the performance of the model.

This technique helps understand how multiple factors affect the target variable and can be used for predictions."""

# Generate data
np.random.seed(42)
data = {
    'Square_Feet': np.random.randint(500, 3000, 100),
    'Num_Rooms': np.random.randint(2, 7, 100),
    'House_Age': np.random.randint(0, 30, 100),
    'Price': np.random.randint(50000, 500000, 100)
}

df = pd.DataFrame(data)

encoded_df = pd.get_dummies(
    df[['Square_Feet','Num_Rooms','House_Age']],
    # columns=df['Num_Rooms'],
    drop_first=True
)

X = sm.add_constant(encoded_df)
Y = df['Price']

# ==================================================split data
train_x,test_x,train_y,test_y = train_test_split(X,Y,train_size=0.8,random_state=33)

df_model1 = sm.OLS(train_y,train_x).fit()
# print(df_model1.summary2()) #! residuals are not normally distributed

# Q-Q plot
stats.probplot(df_model1.resid, dist='norm', plot=plt)
plt.title('Q-Q plot')
# plt.show()      #? Plot looks good

# multi-collinearity
X = df[['Square_Feet','Num_Rooms','House_Age']]
vif_data = pd.DataFrame()
vif_data['Features'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
print(vif_data)     #? since vif<5 all good

# R-squared

pred_y = df_model1.predict(test_x)
print(np.abs(r2_score(test_y,pred_y)))
print(np.sqrt(mean_squared_error(test_y,pred_y)))

# Predict data using the model for a single house
# Predict data using the model for a single house
new_data = pd.DataFrame({
    'Square_Feet': [2000],
    'Num_Rooms': [3],
    'House_Age': [15]
})

# Encode the new data the same way as the training data
encoded_new_data = pd.get_dummies(new_data, drop_first=True)

# Add the constant term (intercept) like in the training data
encoded_new_data = sm.add_constant(encoded_new_data)

# Align the new data columns with the training data
encoded_new_data = encoded_new_data.reindex(columns=train_x.columns, fill_value=0)

# Make prediction
predicted_price = df_model1.predict(encoded_new_data)
print(predicted_price)
