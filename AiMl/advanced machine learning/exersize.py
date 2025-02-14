import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import resample, shuffle
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

earning_df = pd.read_csv('C:\\Users\\My PC\\Desktop\\Internship\\Machine Learning (Codes and Data Files)\\Data\\Earnings Manipulation 220.csv')
# print(earning_df.head(5))
# print(earning_df.info())
# print('earning_df:\n',earning_df['MANIPULATOR'].value_counts())

# ====================================  Under sampling
earning_df_yes = earning_df[earning_df.MANIPULATOR == 1]
earning_df_no = earning_df[earning_df.MANIPULATOR == 0]
balanced = resample(
    earning_df_yes,
    random_state=42,
    n_samples=180
)
balanced_df = pd.concat([earning_df_no,balanced])
# print('balanced_df:\n',balanced_df['MANIPULATOR'].value_counts())
X_features = list(balanced_df.columns)
X_features.remove('MANIPULATOR')
Y = balanced_df.MANIPULATOR
X = pd.get_dummies(balanced_df[X_features],drop_first=True)
train_x,test_x,train_y,test_y = train_test_split(X,Y,random_state=42,test_size=0.2)

# radm_clf = RandomForestClassifier(
#     max_depth=10,n_estimators=15,

#     min_samples_leaf=1,min_samples_split=2,
# )

# radm_clf.fit(train_x,train_y)
# pred_y = radm_clf.predict(test_x)

# print('roc_auc of random forest:',metrics.roc_auc_score(test_y,pred_y))
# print('confusion metrics',metrics.confusion_matrix(test_y,pred_y))

# earning_model = LogisticRegression()
# ada_clf = AdaBoostClassifier()
# ada_clf.fit(train_x,train_y)
# pred_y = ada_clf.predict(test_x)
# print('roc_auc of Ada:',metrics.roc_auc_score(test_y,pred_y))
# print('confusion metrics',metrics.confusion_matrix(test_y,pred_y))

g_clf = GradientBoostingClassifier(n_estimators=20,max_depth=5)
g_clf.fit(train_x,train_y)
pred_y = g_clf.predict(test_x)
print('roc_auc of gradient:',metrics.roc_auc_score(test_y,pred_y))
print('confusion metrics',metrics.confusion_matrix(test_y,pred_y))

