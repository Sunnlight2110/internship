import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sn


hr_df = pd.read_csv('C:\\Users\\My PC\\Desktop\\Internship\\Machine Learning (Codes and Data Files)\\Data\\hr_data.csv')

# print(hr_df.to_string)
# print(hr_df.info())

X_features = list(hr_df.columns)
X_features.remove('Status')
encoded_df = pd.get_dummies(hr_df[X_features],drop_first=True).astype(int)

Y = hr_df['Status'].map(lambda x : int(x == 'Not Joined'))
X = sm.add_constant(encoded_df)

    
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
high_vif_features = vif_data[vif_data["VIF"] > 10]["Feature"].tolist()
X = X.drop(columns=high_vif_features)
print(X.columns)

train_x,test_x,train_y,test_y = train_test_split(X,Y, test_size=0.3, random_state=42)

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)  # Use the same scaler for the test data

hr_model1 = sm.Logit(train_y,train_x).fit(maxiter=12400)
# print(hr_model1.summary2())

def get_significant_vars(model,threshold):
    var_p_values = pd.DataFrame(model.pvalues)
    var_p_values['vars'] = var_p_values.index
    var_p_values.columns = ['pvals','vars']
    return list(var_p_values[var_p_values.pvals <= threshold]['vars'])

significant_vars = get_significant_vars(hr_model1,0.5)
print(significant_vars)


valid_significant_vars = [var for var in significant_vars if var in X.columns]
X_significant = X[valid_significant_vars]
train_y.reset_index(drop=True, inplace=True)
X_significant.reset_index(drop=True, inplace=True)

# Ensure alignment and non-empty DataFrames
X_significant, train_y = X_significant.align(train_y, join='inner', axis=0)
if X_significant.empty or train_y.empty:
    raise ValueError("X_significant or train_y is empty after alignment")

hr_model2 = sm.Logit(train_y, sm.add_constant(X_significant)).fit(maxiter=12400)
print(hr_model2.summary2())
