import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import influence_plot
import matplotlib.pyplot as plt
from scipy import stats

# Set the seed for reproducibility
np.random.seed(42)

# Generate 10th-grade marks (out of 100) between 40 and 100
marks_10th = np.random.uniform(40, 100, 50)

# Define a linear relationship: Salary = m * Marks + c + noise
m = 400  # Slope (how much salary increases per unit of marks)
c = 100  # Intercept (base salary)
noise = np.random.normal(0, 1, 50)  # Adding random noise for variability

Y = m * marks_10th + c + noise  # Salaries based on marks

salary_df = pd.DataFrame({
    '10th_marks':marks_10th,
    'Salary':Y
})

Y = salary_df['Salary']

# Prepare X with a constant term for the intercept
X = sm.add_constant(salary_df['10th_marks'])  #? Adds a constant term (intercept) to the features

#? Split dataset into training and validation sets (80/20 split)
trainx, validx, trainy, validy = train_test_split(X, Y, train_size=0.8, random_state=100)

print(trainx.dtypes)
# print(trainy.dtypes)

# Fit the OLS model
salary_lm = sm.OLS(trainy, trainx).fit()    #fitted data

# Residuals
resid = salary_lm.resid

probplot = sm.ProbPlot(resid)

"""residuals(dots) should be as close as possible to line
if not, your model is not appropriate for data """
fig,ax = plt.subplots()
# probplot.ppplot(line='45',ax = ax)
# ax.set_xlim([-0.5,1.5])
# ax.set_ylim([-0.5,1.5])
# plt.title('residues')
# plt.show()

# def get_standardized_value(values):
#     return (values - values.mean())/values.std()
# plt.scatter(get_standardized_value(salary_lm.fittedvalues), get_standardized_value(resid))
"""
No pattern(should be)
    scattered around the horizontal zero line, with no clear pattern, it indicates that your model is doing well!
    Ideal Scenario: No pattern and a uniform spread.

    There is no systematic bias (no tendency for the residuals to be larger for certain predictions).
    The model is appropriately capturing the relationship between marks and salary.
    The residuals are normally distributed (a prerequisite for OLS regression).

Funnel Shape or Spread Increasing/Decreasing
    residuals get larger as the fitted values increase or decrease), it indicates heteroscedasticity:

    This means that the variance of the residuals is not constant across the range of fitted values.
    In simpler terms, the model`s errors are more variable at certain points 

    What to do: If you see this, it may suggest the need for a transformation of the dependent variable (like log-transformation) 
        or you may need to reconsider the model choice (maybe nonlinear regression or robust regression).

Curved or Non-Linear Pattern
     it suggests that the model is missing a nonlinear relationship between the independent variable (marks) and the dependent variable (salary).

     What to do: If you see a curved pattern, it might be worth considering:
        Polynomial regression (adding higher-order terms like marks²).
        Nonlinear regression models.

Clusters or Outliers:
     If you see certain points far away from the general cluster (outliers), this indicates outliers or high-leverage points in the data.

     What to do: Investigate outliers to see if they are genuine or errors in the data. If they are influential points, 
        you might want to consider removing them or using a more robust regression method.
"""

# plt.title('Residual plot')
# plt.xlabel('standardize predicted value')
# plt.ylabel('standardize residuals')
# plt.grid(True)
# plt.show()


# ? outlier analysis
# ? Z test

"""steps
    Compute the mean and standard deviation of your dataset.
    Calculate the Z-score for each data point.
    Set a threshold (commonly 3 or -3). Any Z-score beyond this range is considered an outlier.
    """

salary_df['z_score_salary'] = stats.zscore(salary_df.Salary)
# salary_df[(salary_df.z_score_salary>3.0) | (salary_df.z_score_salary<-3.0)]

# plt.figure(figsize=(10,6))
# plt.scatter(salary_df['10th_marks'], salary_df['Salary'], label='Data points')  #Original data
# plt.scatter(salary_df.loc[(salary_df.z_score_salary > 3.0) | (salary_df.z_score_salary < -3.0)]['10th_marks'],  #Outliers data
#             salary_df.loc[(salary_df.z_score_salary > 3.0) | (salary_df.z_score_salary < -3.0)]['Salary'],
#             color='red', label='Outliers')

# plt.title('Salary vs 10th Marks with Outliers')
# plt.xlabel('10th Marks')
# plt.ylabel('Salary')
# plt.legend()
# plt.show()

# ? cooks distance
# influence = salary_lm.get_influence()
# (c,p) = influence.cooks_distance
# plt.stem(np.arange(len(trainx)),np.round(c,3),markerfmt=",")
# plt.title('Cooks distance for all observation')
# plt.xlabel('Row index')
# plt.ylabel('cooks distance')
# # plt.show()

# # ? leverage values
# fig , ax = plt.subplots(figsize = (8,6))
# influence_plot(salary_lm, ax = ax)
# plt.title('leverage value vs residuals')
# plt.show()


# ? predict using validation dataset

pred_y = salary_lm.predict(validx)

# ? checking accuracy of model

print(np.abs(r2_score(validy,pred_y)))

print(np.sqrt(mean_squared_error(validy,pred_y)))   #sqrt = square root

#? predict low and high intervals for y
"""In the context of linear regression, wls_prediction_std stands for Weighted Least Squares Prediction Standard Error. 
    It`s a function that provides the standard error of the predictions made by a weighted least squares (WLS) regression model.
    
    Lower wls = batter"""
_,pred_y_low,pred_y_high = wls_prediction_std(salary_lm,validx,alpha = 0.1)

#  store values in dataframe
pred_y_df = pd.DataFrame({
    'grade_10' : validx['10th_marks'],
    'pred_y_lef' : pred_y_low,
    'pred_y_right' : pred_y_high
})

#? pred_y_df is our prediction dataset

# predict single
mark = np.array([80,0])
X = sm.add_constant(mark)  # 2D array (1 row, 1 feature + constant)
pred = salary_lm.predict(X)
print(f"Marks {mark[0]}: Predicted Salary {pred[0]}")



