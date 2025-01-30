import numpy as np
import pandas as pd
import scipy as sm
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


ipl_auction_df = pd.read_csv( 'C:\\Users\\My PC\\Desktop\\Internship\\Machine Learning (Codes and Data Files)\\Data\\IPL IMB381IPL2013.csv' ) 
X_features = ipl_auction_df.columns 
X_features = ['AGE', 'COUNTRY', 'PLAYING ROLE',
'T-RUNS', 'T-WKTS', 'ODI-RUNS-S', 'ODI-SR-B',
'ODI-WKTS', 'ODI-SR-BL', 'CAPTAINCY EXP', 'RUNS-S',
'HS', 'AVE', 'SR-B', 'SIXERS', 'RUNS-C', 'WKTS',
'AVE-BL', 'ECON', 'SR-BL']

ipl_auction_df['PLAYING ROLE'].unique() 
pd.get_dummies(ipl_auction_df['PLAYING ROLE'])[0:5]

categorical_features = ['AGE', 'COUNTRY', 'PLAYING ROLE', 'CAPTAINCY EXP'] 
ipl_auction_encoded_df = pd.get_dummies( ipl_auction_df[X_features],
columns = categorical_features,
drop_first = True ) 
ipl_auction_encoded_df.columns
X_features = ipl_auction_encoded_df.columns 
X = sm.add_constant( ipl_auction_encoded_df )
Y = ipl_auction_df['SOLD PRICE']
train_X, test_X, train_y, test_y = train_test_split( X ,
Y,
train_size = 0.8,
random_state = 42 ) 
ipl_model_1 = sm.OLS(train_y, train_X).fit() 

