import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

store_df = pd.read_excel('C:\\Users\\My PC\\Desktop\\Internship\\Machine Learning (Codes and Data Files)\\Data\\store.xls')
# print('store_df.head(5)\n',store_df.head(5))
# print('store_df.info()\n',store_df.info())

store_df.set_index(
    pd.to_datetime(store_df.Date),inplace=True  #datetime converts Date column into datetime series, set_index = sets Date as index
)
store_df.drop('Date',axis=1, inplace=True)
store_df.index.freq = 'D'

# print(store_df[:7])

# plt.figure(figsize=(10,4))
# plt.xlabel('Date')
# plt.ylabel('Demand')
# plt.plot(store_df.demand)
# plt.show()

# acf_plot = plot_acf(store_df.demand, lags = 20)
# plt.show()

def adfuller_test(ts):
    adfuller_result = adfuller(ts, autolag=None)  # Apply the ADF test
    adfuller_out = pd.Series(
        adfuller_result[0:4],  # Extracting first 4 results
        index = [
            'Test Statistic',     # ADF test statistic
            'p-value',            # The p-value associated with the test statistic
            'Lags Used',          # Number of lags used in the test
            'Number of Observations Used'  # Number of observations used in the test
        ]
    )
    print('adfuller_out', adfuller_out)  # Output the result

# adfuller_test(store_df.demand)

# Differencing
# ?converts non stationary data into stationary
store_df['demand_diff'] = store_df.demand - store_df.demand.shift(2)
# print('first difference',store_df.head(5))
store_df_diff = store_df.dropna()
# print(adfuller_test(store_df_diff.demand_diff))

# plt.figure(figsize=(10,4))
# plt.xlabel('Date')
# plt.ylabel('First Order Diffrence')
# plt.plot(store_df_diff.demand_diff)
# plt.show()

# pacf_plot = plot_pacf(
#     store_df_diff.demand_diff, lags = 10
# )
# print(adfuller_test(store_df_diff.demand_diff))
store_train = store_df[0:100]
store_test = store_df[100:]
# plt.show()

arima = ARIMA(
    store_train.demand.astype(np.float64),
    order = (1,1,1)
)
arima_model = arima.fit(start_params=[0.1,0.1,1])
# print(arima_model.summary())

# acf_plot = plot_acf(arima_model.resid, lags = 20)
# pacf_plot = plot_pacf(arima_model.resid, lags = 20)
# plt.show()

store_predict = arima_model.forecast(steps = 15)
# print(store_predict)

def get_mape(actual, predicted):
    test_y, pred_y = np.array(actual),np.array(predicted)
    return np.round(np.mean(np.abs(
        (actual-predicted) / actual
    ))*100,2)
print(get_mape(
    store_df.demand[100:],
    store_predict
))