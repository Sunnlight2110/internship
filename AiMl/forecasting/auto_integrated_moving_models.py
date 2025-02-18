import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

vimana_df = pd.read_csv('C:\\Users\\My PC\\Desktop\\Internship\\Machine Learning (Codes and Data Files)\\Data\\vimana.csv')
# print('vimana_df.head(5)\n',vimana_df.head(5))
# print('vimana_df.info()\n',vimana_df.info())

# acf_plot = plot_acf(
#     vimana_df.demand, lags=20
# )
# plt.show()

# pacf_plot = plot_pacf(
#     vimana_df.demand,
#     lags = 18
# )
# plt.show()

arima = ARIMA(vimana_df.demand[0:30].astype(np.float64), order = [1,0,0])
ar_model = arima.fit()
# print(ar_model.summary())

def get_mape(actual, predicted):
    test_y, pred_y = np.array(actual),np.array(predicted)
    return np.round(np.mean(np.abs(
        (actual-predicted) / actual
    ))*100,2)

forecast_31_37 = ar_model.predict(30,36)    #(start,end)
# print(forecast_31_37)
print('MAPE for 31 to 37',get_mape(
    vimana_df.demand[30:],
    forecast_31_37
))

# Moving average(MA)
arima = ARIMA(
    vimana_df.demand[0:30].astype(np.float64),
    order = (0,0,1)
)
ma_model = arima.fit()
# print('MA summary\n',ma_model.summary())
forecast_31_37 = ma_model.predict(30,36)
print('MAPE for MA',get_mape(
    vimana_df.demand[30:],
    forecast_31_37
))

# ARMA model(Auto regression moving average)
arma = ARIMA(
    vimana_df.demand[:30].astype(np.float64),
    order = (1,0,1)
)
arma_model = arma.fit()
print('arma model summary\n',arma_model.summary())
forecast_31_37 = arma_model.predict(30,36)
print('MAPE for ARMA',get_mape(
    vimana_df.demand[30:],
    forecast_31_37
))