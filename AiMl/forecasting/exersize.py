import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


nuclier_df = pd.read_csv('C:\\Users\\My PC\\Desktop\\Internship\\Machine Learning (Codes and Data Files)\\Data\\Nuclear Capacity.csv')
# print(nuclier_df.head(5))

# ?convert into date time formate
nuclier_df['date'] = nuclier_df['date'].map(lambda x : x[0:4]+'/'+x[5:7]+'/'+x[7:9])
nuclier_df['date'] = pd.to_datetime(nuclier_df.date)
nuclier_df.set_index(nuclier_df.date,inplace=True)
nuclier_df.index = pd.date_range(start=nuclier_df.index.min(), end=nuclier_df.index.max(), freq='D')
nuclier_df.drop('date',axis=1, inplace=True)
nuclier_df = nuclier_df.asfreq('D')


nuclier_monthly_df = nuclier_df.resample('ME').last()
# print(nuclier_monthly_df.head(10))

# # ?remove down spike(outliers)
index=nuclier_monthly_df.index[nuclier_monthly_df.Megawatts < 90000]
nuclier_monthly_df.drop(index,axis=0, inplace=True)

# plt.plot(nuclier_monthly_df.Megawatts)
# plt.xlabel('date')
# plt.ylabel('Consumption(Megawatts)')
# plt.show()
# decomposition = seasonal_decompose(nuclier_monthly_df['Megawatts'], model='additive',period=60)
# decomposition.plot()
# plt.show()

# # ?ACF and PACF
# plot_acf(nuclier_monthly_df['Megawatts'], lags=40)
# plot_pacf(nuclier_monthly_df['Megawatts'], lags=40)
# plt.show()

# # ?differencing data
nuclier_monthly_df['Megawatts_diff'] = nuclier_monthly_df.Megawatts - nuclier_monthly_df.Megawatts.shift(2)
nuclier_monthly_df.dropna(inplace=True)
# print(nuclier_monthly_df)
# plot_acf(nuclier_monthly_df['Megawatts_diff'], lags=20)
# plot_pacf(nuclier_monthly_df['Megawatts_diff'], lags=20)
# plt.show()
period = 80

# ?animation for seasonal decomposition
# fig,(ax1,ax2,ax3,ax4) = plt.subplots(4,1,figsize=(12,10))

# def update(val):
#     period = int(slider.val)
#     result = seasonal_decompose(nuclier_monthly_df['Megawatts_diff'], model='additive',period=period)

#     # clear pervious axis
#     ax1.clear()
#     ax2.clear()
#     ax3.clear()
#     ax4.clear()

#     ax1.plot(nuclier_monthly_df['Megawatts_diff'], label='original data',color='blue')
#     ax1.legend(loc='upper left')
#     ax1.set_title(f'Original Data (Period {period})')

#     ax2.plot(result.trend, label='trend',color='orange')
#     ax2.legend(loc='upper left')
#     ax2.set_title(f'trend (Period {period})')

#     ax3.plot(result.seasonal, label='seasonal',color='orange')
#     ax3.legend(loc='upper left')
#     ax3.set_title(f'seasonal (Period {period})')

#     ax4.plot(result.resid, label='resid',color='orange')
#     ax4.legend(loc='upper left')
#     ax4.set_title(f'resid (Period {period})')

# ax_slider = plt.axes([0.15,0.01,0.7,0.05],facecolor = 'green')
# slider = Slider(ax_slider, 'Period', 1,period, valinit=period, valstep=1)

# slider.on_changed(update)
# plt.subplots_adjust(hspace=0.4)
# plt.show()

srima_model = SARIMAX(
    nuclier_monthly_df['Megawatts_diff'], 
    order = (1,1,1), seasonal_order=(1,1,1,12),
    enforce_invertibility=False, enforce_stationarity=False
)
sarima_result = srima_model.fit()
# print(sarima_result.summary())
train_size = int(len(nuclier_monthly_df)*0.8)
train = nuclier_monthly_df.iloc[:train_size]
test = nuclier_monthly_df.iloc[train_size:]

forecast = sarima_result.predict(start=len(train), end=len(train+test)-1, dynamic=False)
plt.plot(test.index, test['Megawatts_diff'], label='actual',color='blue')
plt.plot(test.index, forecast, label='predicted',color='red')
plt.legend()
plt.title('Actual vs predictions')
plt.show()

print('R squared:',r2_score(test['Megawatts_diff'],forecast))
print('MSE:',mean_squared_error(test['Megawatts_diff'],forecast))
print('RMSE:',np.sqrt(mean_squared_error(test['Megawatts_diff'],forecast)))