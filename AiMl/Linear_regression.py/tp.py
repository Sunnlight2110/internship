import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from scipy import stats
from statsmodels.graphics.regressionplots import influence_plot


# Set seed for reproducibility
np.random.seed(42)

# Generate random sizes of plots (in square feet) between 500 and 3500
sizes = np.random.uniform(500, 3500, 100)

# Define a linear relationship: Price = m * Size + c + noise
m = 150  # slope (price increases by 150 for each additional square foot)
c = 50000  # intercept (base price)
noise = np.random.normal(0, 1, 100)  # random noise for variability

prices = m * sizes + c + noise  # Prices based on the size
df = pd.DataFrame({
    'Size' : sizes,
    'Price' : prices
})
X = sm.add_constant(df['Size'])
Y = df['Price']

trainx,validx,trainy,validy = train_test_split(X,Y, train_size = 0.8, random_state = 100)

price_lm = sm.OLS(trainy,trainx).fit()  #Fitted data

# # Check residuals
resid = price_lm.resid
fig,ax = plt.subplots()
proplot = sm.ProbPlot(resid)
proplot.ppplot(line='45',ax = ax)
ax.set_xlim([-0.5,1.5])
ax.set_ylim([-0.5,1.5])
plt.title('residuals')
plt.show()

def get_standerdize_value(value):
    return (value - value.mean()) / value.std()
plt.scatter(get_standerdize_value(price_lm.fittedvalues),get_standerdize_value(resid))
plt.title('Residual plot')
plt.xlabel('standardize predicted value')
plt.ylabel('standardize residuals')
plt.grid(True)
plt.show()


# Z test:
df['z_score_price'] = stats.zscore(df.Price)
plt.figure(figsize=(10,6))
plt.scatter(df['Price'],df['Size']) #Original data
plt.scatter(
    df.loc[(df.z_score_price > -3.0) & (df.z_score_price < 3.0)]['Price'],
    df.loc[(df.z_score_price > -3.0) & (df.z_score_price < 3.0)]['Size']
)
plt.title('Z score')
plt.xlabel('Price')
plt.ylabel('Size')
plt.show()

# Cooks distance
influence = price_lm.get_influence()
(c, p) = influence.cooks_distance
plt.stem(np.arange(len(trainx)),np.round(c,3),markerfmt=',')
plt.title('Cooks distance for all observation')
plt.xlabel('Row index')
plt.ylabel('cooks distance')
plt.show()

# leverage values
fig , ax = plt.subplots(figsize = (8,6))
influence_plot(price_lm, ax = ax)
plt.title('leverage value vs residuals')
plt.show()

pred_y = price_lm.predict(validx)

#  Predict using valid dataset
print(np.abs(r2_score(validy,pred_y)))
print(np.sqrt(mean_squared_error(validy,pred_y)))

# predict low and high intervals for y
_,pred_y_low,pred_y_high = wls_prediction_std(price_lm,validx,alpha = 0.1)
pred_y_df = pd.DataFrame({
    'Size' : validx['Size'],
    'pred_y_lef' : pred_y_low,
    'pred_y_right' : pred_y_high
})
print(pred_y_low,pred_y_high)