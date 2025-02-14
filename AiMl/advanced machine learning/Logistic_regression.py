import pandas as pd
import numpy as np
from sklearn.utils import resample, shuffle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import seaborn as sn
import matplotlib.pyplot as plt
bank_df = pd.read_csv('C:\\Users\\My PC\\Desktop\\Internship\\Machine Learning (Codes and Data Files)\\Data\\bank.csv')
# print(bank_df.info())
# print(bank_df.subscribed.value_counts())

bank_subscribed_no = bank_df[bank_df.subscribed == 'no']
bank_subscribed_yes = bank_df[bank_df.subscribed == 'yes']

df_minority_oversample =  resample(
    bank_subscribed_yes,
    replace= True,
    n_samples=2000
)

new_bank_df = pd.concat([bank_subscribed_no,df_minority_oversample])

new_bank_df = shuffle(new_bank_df)
# print('old df:\n',bank_df.head(5))
# print('new df:\n',new_bank_df.head(5))
# print(new_bank_df.subscribed.value_counts())

X_features = list(new_bank_df.columns)
X_features.remove('subscribed')

encoded_df = pd.get_dummies(new_bank_df[X_features],drop_first=True)
Y = new_bank_df.subscribed.map(lambda x : int(x == 'yes'))
scale = StandardScaler()
X = scale.fit_transform(encoded_df)

train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.3,random_state=42)

bank_model1 = LogisticRegression(max_iter=2000,C = 1)
bank_model1.fit(train_x,train_y)

pred_y = bank_model1.predict(test_x)

def draw_cm(actual,predicted):
    cm = metrics.confusion_matrix(actual,predicted, labels=[1,0])
    sn.heatmap(
        cm, annot=True, fmt='.2f',
        xticklabels=['subscribed','Not subscriber'],
        yticklabels=['subscribed','Not subscriber']
    )

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# cm = draw_cm(test_y,pred_y)
# print(metrics.classification_report(test_y,pred_y))

predicted_df = pd.DataFrame(bank_model1.predict_proba(test_x))
# print(predicted_df.head(10))

test_result_df = pd.DataFrame({'actual':test_y})
test_result_df = test_result_df.reset_index()
test_result_df['chd1'] = predicted_df.iloc[:,1:2]
# print(test_result_df.head(5))

auc_score = metrics.roc_auc_score(test_result_df.actual, test_result_df.chd1)
# print('auc score: ',round(float(auc_score),2))


knn_clf = KNeighborsClassifier()
knn_clf.fit(train_x,train_y)
pred_y = knn_clf.predict(test_x)
# draw_cm(test_y,pred_y)
# print(metrics.confusion_matrix(test_y,pred_y))
# print(metrics.classification_report(test_y,pred_y))

tuned_parameters = [{
    'n_neighbors': range(5,10),
    'metric': ['canberra','euclidean','minkowski']
}]

clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv = 10, scoring='roc_auc')
clf.fit(train_x,train_y)
# print('best score',clf.best_score_)
# print('best parameters',clf.best_params_)

radm_clf = RandomForestClassifier(max_depth=10, n_estimators=10)
radm_clf.fit(train_x,train_y)
RandomForestClassifier(
    max_depth=15, max_features='auto', max_leaf_nodes=None, 
    min_impurity_decrease=0.0,
    min_samples_leaf=1, min_samples_split=2, 
    min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=None, 
    oob_score=False, random_state=None, verbose=0, 
    warm_start=False
)
"""
RandomForestClassifier with max_depth=15 (limits tree depth), max_features='sqrt' (best feature selection), 
max_leaf_nodes=None (unlimited leaves), min_impurity_decrease=0.0 (no forced pruning), 
min_samples_leaf=1 (smallest leaf size), min_samples_split=2 (splits with at least 2 samples), 
min_weight_fraction_leaf=0.0 (no weighted constraints), n_estimators=20 (number of trees), 
n_jobs=None (single-threaded), oob_score=False (no out-of-bag validation), 
random_state=None (random results), verbose=0 (silent mode), warm_start=False (no incremental training).
"""

tuned_parameters = [{
    'max_depth' : [109,15],
    'n_estimators' : [10,20],
    'max_features' : ['sqrt', 0.2]
}]
radm_clf = RandomForestClassifier()
clf = GridSearchCV(
    radm_clf,
    tuned_parameters,
    cv = 5,
    scoring='roc_auc'
)
clf.fit(train_x,train_y)
# print('best score',clf.best_score_)
# print('best parameters',clf.best_params_)

radm_clf.fit(train_x,train_y)
pred_y = radm_clf.predict(test_x)
# print('confution matrix',metrics.confusion_matrix(test_y,pred_y))
# print('classification report',metrics.classification_report(test_y,pred_y))

feature_rank = pd.DataFrame({
    'feature' : encoded_df.columns,
    'importance' : radm_clf.feature_importances_,
})

feature_rank = feature_rank.sort_values('importance',ascending=False)
# plt.figure(figsize=(8,6))
# sn.barplot(y='feature',x='importance',data = feature_rank)
# plt.show()  

bank_clf = LogisticRegression()
ada_clf = AdaBoostClassifier(bank_clf,n_estimators=50)
ada_clf.fit(train_x,train_y)

AdaBoostClassifier(
    algorithm='SAMME.R',
    estimator=LogisticRegression(
        C=1, class_weight=None, dual = False, fit_intercept= True,
        intercept_scaling=1, max_iter=100, multi_class='ovr',
        n_jobs=1, penalty='l2',random_state=None, solver='liblinear',
        tol=0.0001, verbose=0, warm_start=False
    ),
    learning_rate=1,n_estimators=200,random_state=None
)

"""algorithm='SAMME.R' → Uses real-valued boosting (probability-based).
2️⃣ estimator=LogisticRegression(...) → Uses Logistic Regression as the weak model.
3️⃣ learning_rate=1 → Controls how much each weak model contributes (default = 1).
4️⃣ n_estimators=50 → Number of weak models (trees or regressors) being trained.
5️⃣ random_state=None → No fixed randomness (use a value for reproducibility).

Logistic Regression Parameters (Inside estimator)
1️⃣ C=1 → Regularization strength (higher = less regularization).
2️⃣ fit_intercept=True → Adds an intercept (bias term).
3️⃣ max_iter=100 → Maximum iterations for convergence.
4️⃣ multi_class='ovr' → One-vs-Rest strategy (binary classifiers for multi-class).
5️⃣ penalty='l2' (L2 regularization for better generalization).
6️⃣ solver='liblinear' → Good for small datasets & L1/L2 penalty.
7️⃣ tol=0.0001 → Small tolerance for stopping criterion."""
pred_y = ada_clf.predict(test_x)

# draw_cm(test_y,pred_y)
# print('confution matrix\n',metrics.confusion_matrix(test_y,pred_y))
# print('classification report\n',metrics.classification_report(test_y,pred_y))

g_boost_clf = GradientBoostingClassifier(
    n_estimators=500,
    max_depth=10
)

g_boost_clf.fit(train_x,train_y)
g_boost_clf.predict(test_x)

# print('confution matrix\n',metrics.confusion_matrix(test_y,pred_y))
# print('classification report\n',metrics.classification_report(test_y,pred_y))

cv_score = cross_val_score(
    g_boost_clf, train_x, train_y,
    cv=10, scoring='roc_auc'
)

print('cv_score',cv_score)
print(f'Mean accuracy:{np.mean(cv_score)} with std of:{np.std(cv_score)}')