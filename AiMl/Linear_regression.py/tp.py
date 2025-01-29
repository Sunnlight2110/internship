import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

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

# Check residuals
resid = price_lm.resid
fig,ax = plt.subplots()
proplot = sm.ProbPlot(resid)
proplot.ppplot(line='45',ax = ax)
ax.set_xlim([-0.5,1.5])
ax.set_ylim([-0.5,1.5])
plt.title('residuals')
plt.show()
