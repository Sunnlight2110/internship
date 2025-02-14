import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import metrics
ipl_df = pd.read_csv('C:\\Users\\My PC\\Desktop\\Internship\\Machine Learning (Codes and Data Files)\\Data\\IPL IMB381IPL2013.csv')
# print(ipl_df.info())

X_features = [
    'AGE',
    'COUNTRY',
    'PLAYING ROLE',
    'T-RUNS',
    'T-WKTS',
    'ODI-RUNS-S',
    'ODI-SR-B',
    'ODI-WKTS',
    'ODI-SR-BL',
    'CAPTAINCY EXP',
    'RUNS-S',
    'HS',
    'AVE',
    'SR-B',
    'SIXERS',
    'RUNS-C',
    'WKTS',
    'AVE-BL',
    'ECON',
    'SR-BL',
]

categorial_features = ['AGE','COUNTRY','PLAYING ROLE','CAPTAINCY EXP']

encoded_df = pd.get_dummies(
    ipl_df[X_features],
    columns= categorial_features,
    drop_first= True
)
# print(encoded_df.columns)

X = encoded_df
Y = ipl_df['SOLD PRICE']

X_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)
# standardize Y
Y = (Y-Y.mean())/Y.std()

# split data
train_x,test_x,train_y,test_y = train_test_split(X_scaled,Y,test_size=0.2,random_state=42)

ipl_lin_model = LinearRegression()
ipl_lin_model.fit(train_x,train_y)
# print(ipl_lin_model.coef_)

columns_coef_df = pd.DataFrame({
    'columns' : encoded_df.columns,
    'coef' : ipl_lin_model.coef_
})
# print(columns_coef_df)
sorted_coef = columns_coef_df.sort_values('coef', ascending=False)

# plt.figure(figsize=(8,6))
# sn.barplot(x='coef',y='columns',data=sorted_coef)
# plt.xlabel('coefficients from linear regression')
# plt.ylabel('Features')
# plt.show()
def get_train_test_rmse(model):
    train_pred_y = model.predict(train_x)
    rmse_train = round(np.sqrt(metrics.mean_squared_error(train_y,train_pred_y)),3)
    test_pred_y = model.predict(test_x)
    rmse_test = round(np.sqrt(metrics.mean_squared_error(test_y,test_pred_y)),3)
    print(f"RSME for {model} = train:{rmse_train}, test:{rmse_test}")

get_train_test_rmse(ipl_lin_model)

# K = 5
# kf = KFold(n_splits=K,shuffle=True,random_state=42)
# # print(kf)
# train_rsme = []
# test_rsme = []
# for train_index,test_index in kf.split(X):
#     train_x,test_x = X.iloc[train_index], X.iloc[test_index]
#     train_y,test_y = Y.iloc[train_index],Y.iloc[test_index]

#     ipl_model2 = LinearRegression()
#     ipl_model2.fit(train_x,train_y)

#     y_train_pred = ipl_model2.predict(train_x)
#     y_test_pred = ipl_model2.predict(test_x)

#     train_rsme.append(np.sqrt(metrics.mean_squared_error(train_y, y_train_pred)))
#     test_rsme.append(np.sqrt(metrics.mean_squared_error(test_y, y_test_pred)))

# # print('RMSE for K')
# # print(f'Average Train RMSE: {np.mean(train_rsme):.3f}')
# # print(f'Average Test RMSE: {np.mean(test_rsme):.3f}')

#? ridge
ridge = Ridge( alpha= 2.0, max_iter=1000)
ridge.fit(train_x,train_y)
get_train_test_rmse(ridge)

# ? Lasso
lasso = Lasso(alpha = 0.01, max_iter=500)
lasso.fit(train_x,train_y)
get_train_test_rmse(lasso)

# storing features and coefficients in lasso
lasso_coef_df = pd.DataFrame({
    'columns' : encoded_df.columns,
    'coef_' : lasso.coef_
})
# print("lasso_coef_df\n",lasso_coef_df)

# ? ElasticNet = lasso + ridge
enet = ElasticNet(alpha= 1.01, l1_ratio=0.001, max_iter=500)
enet.fit(train_x,train_y)
get_train_test_rmse(enet)