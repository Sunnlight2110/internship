import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

wsb_df = pd.read_csv('C:\\Users\\My PC\\Desktop\\Internship\\Machine Learning (Codes and Data Files)\\Data\\wsb.csv')
# print(wsb_df.head(10))
# print(wsb_df.info())

# plt.figure(figsize=(10,4))
# plt.xlabel('months')
# plt.ylabel('Quantity')
# plt.plot(wsb_df['Sale Quantity'])
# plt.show()

wsb_df['mavg_12'] = wsb_df['Sale Quantity'].rolling(window=12).mean().shift(1)

"""
The rolling() function is applied to a series (like a column in a DataFrame).
    It creates a moving window of 12 rows (because window=12).
The shift() function shifts the values in the column by a specified number of periods.
    In this case, .shift(1) means the values in the column will be shifted down by 1 position.        
"""

pd.set_option(
    'display.float_format',
    lambda x : '%0.2f' % x
)
# print(wsb_df[['Sale Quantity','mavg_12']][36:])

# plt.figure(figsize=(10,4))
# plt.xlabel('months')
# plt.ylabel('Quantity')
# plt.plot(wsb_df['Sale Quantity'])
# plt.plot(wsb_df['mavg_12'][12:],'.')
# plt.show()

# Accuracy
def get_mape(actual, predicted):
    test_y, pred_y = np.array(actual),np.array(predicted)
    return np.round(np.mean(np.abs(
        (actual-predicted) / actual
    ))*100,2)

print('mape for moving average smoothing',get_mape(wsb_df['Sale Quantity'][36:].values,wsb_df['mavg_12'][36:].values))


# ? Exponential smoothing
wsb_df['ewm'] = wsb_df['Sale Quantity'].ewm(alpha=0.2).mean()   #high alpha = high weighage to recent observations

pd.options.display.float_format = '{:.2f}'.format
# print(wsb_df)

print('mape for Exponential smoothing',get_mape(
    wsb_df['Sale Quantity'][36:].values,
    wsb_df['ewm'][36:].values
))

# plt.figure(figsize=(10,4))
# plt.xlabel('months')
# plt.ylabel('Quantity')
# plt.plot(wsb_df['Sale Quantity'])
# plt.plot(wsb_df['ewm'][12:],'-')
# plt.show()

ts_decompose = seasonal_decompose(
    np.array(wsb_df['Sale Quantity']),
    model='multiplicative',
    period = 12
)
# ts_plot = ts_decompose.plot()
# wsb_df['seasonal'] = ts_decompose.seasonal
# wsb_df['trend'] = ts_decompose.trend

# plt.show()
