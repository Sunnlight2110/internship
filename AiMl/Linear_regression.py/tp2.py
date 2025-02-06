import warnings
warnings.filterwarnings('ignore') 
import pandas as pd
import numpy as np
credit_df = pd.read_csv('C:\\Users\\My PC\\Desktop\\Internship\\Machine Learning (Codes and Data Files)\\Data\\German Credit Data.csv')
credit_df.dropna()
credit_df.info() 
credit_df.iloc[0:5,1:7] 
credit_df.iloc[0:5,7:]
credit_df.status.value_counts() 
X_features = list( credit_df.columns ) 
X_features.remove( 'status' ) 
X_features
encoded_credit_df = pd.get_dummies( credit_df[X_features], 
drop_first = True )
list(encoded_credit_df.columns)

import statsmodels.api as sm
Y = credit_df.status 
X = sm.add_constant( encoded_credit_df ) 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, 
Y, 
test_size = 0.3, 
random_state = 42) 
X_test = X_test .astype(int)

import statsmodels.api as sm
logit = sm.Logit(y_train, X_train) 
logit_model = logit.fit() 