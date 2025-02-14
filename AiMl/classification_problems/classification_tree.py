# Importing necessary libraries and suppressing warnings
import warnings 
warnings.filterwarnings('ignore') 
import math
import pandas as pd 
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt 
import seaborn as sn 
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus as pdot
from IPython.display import Image
from graphviz import Source
import os
os.environ["PATH"] += os.pathsep + r"C:\Program Files (x86)\Graphviz\bin"

# Reading the dataset
credit_df = pd.read_csv('C:\\Users\\My PC\\Desktop\\Internship\\Machine Learning (Codes and Data Files)\\Data\\German Credit Data.csv')

# Extracting feature columns (excluding 'status' which is the target variable)
X_features = list(credit_df.columns) 
X_features.remove('status')
encoded_credit_df = pd.get_dummies(credit_df[X_features], drop_first=True).astype(int)

Y = credit_df.status 
X = sm.add_constant(encoded_credit_df)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

logit = sm.Logit(y_train, X_train)
logit_model = logit.fit()

def get_significant_vars(lm):
    var_p_vals_df = pd.DataFrame(lm.pvalues)
    var_p_vals_df['vars'] = var_p_vals_df.index
    var_p_vals_df.columns = ['pvals', 'vars']
    # Returning the list of variables with p-value <= 0.05
    return list(var_p_vals_df[var_p_vals_df.pvals <= 0.05]['vars'])

significant_vars = get_significant_vars(logit_model)
final_logit = sm.Logit(y_train, sm.add_constant(X_train[significant_vars])).fit()

y_pred_df = pd.DataFrame({
    "actual": y_test,  # Actual target values
    "predicted_prob": final_logit.predict(sm.add_constant(X_test[significant_vars]))  # Predicted probabilities
})

y_pred_df['predicted'] = y_pred_df.predicted_prob.map(lambda x: 1 if x > 0.5 else 0)
# print(y_pred_df.sample(10, random_state=42))

# Function to plot the confusion matrix
def draw_cm(actual, predicted):
    """
    This function takes in the actual and predicted labels, and displays a confusion matrix as a heatmap.
    """
    cm = metrics.confusion_matrix(actual, predicted, labels=[1, 0])
    # sn.heatmap(cm, annot=True, fmt='.2f', xticklabels=["Bad credit", "Good Credit"], yticklabels=["Bad credit", "Good Credit"])
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')

# draw_cm(y_pred_df.actual, y_pred_df.predicted)
# print(metrics.classification_report(y_pred_df.actual, y_pred_df.predicted))

# Function to plot the ROC curve
def draw_roc(actual, probs):
    """
    This function calculates and plots the Receiver Operating Characteristic (ROC) curve, 
    and returns the false positive rate (fpr), true positive rate (tpr), and thresholds used.
    """
    """
    The threshold is the decision boundary for classifying a prediction as positive (1) or negative (0).
        If probability ≥ threshold → classified as positive (1).
        If probability < threshold → classified as negative (0).
    TPR
        Measures how many actual positives were correctly classified.
        Higher TPR = fewer false negatives (FN).
        If TPR = 1, the model correctly detects all positive
    FPR
        Measures how many actual negatives were wrongly classified as positives.
        Higher FPR = more false positives (bad for models).
        If FPR = 0, the model never misclassifies negatives as positives.
    """
    fpr, tpr, thresholds = metrics.roc_curve(actual, probs, drop_intermediate=False)
    auc_score = metrics.roc_auc_score(actual, probs)
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    # plt.ylabel('True Positive Rate')
    # plt.legend(loc="lower right")
    return fpr, tpr, thresholds

# Draw the ROC curve and get the fpr, tpr, thresholds
fpr, tpr, thresholds = draw_roc(y_pred_df.actual, y_pred_df.predicted_prob)

# Creating a DataFrame for TPR and FPR with their respective thresholds
tpr_fpr = pd.DataFrame({
    'tpr': tpr,
    'fpr': fpr,
    'thresholds': thresholds
})
# Adding a 'diff' column to see the difference between TPR and FPR for each threshold
tpr_fpr['diff'] = tpr_fpr.tpr - tpr_fpr.fpr
# print(tpr_fpr.sort_values('diff', ascending=False)[0:5])

# Applying a new threshold (0.22) for prediction and checking performance
y_pred_df['predicted_new'] = y_pred_df.predicted_prob.map(lambda x: 1 if x > 0.22 else 0)

# Drawing the confusion matrix for the new threshold
draw_cm(y_pred_df.actual, y_pred_df.predicted_new)
# print(metrics.classification_report(y_pred_df.actual, y_pred_df.predicted_new))

# Function to calculate total cost based on false positives and false negatives
def get_total_cost(actual, predicted, cost_FPs, cost_FNs):
    """
    This function calculates the total cost of predictions by multiplying the number of false positives 
    by the cost of false positives, and false negatives by the cost of false negatives.
    """
    cm = metrics.confusion_matrix(actual, predicted, labels=[1, 0])
    cm_mat = np.array(cm)
    return cm_mat[0, 1] * cost_FNs + cm_mat[1, 0] * cost_FPs

# Creating a DataFrame to track the cost for different probability cut-off values
cost_df = pd.DataFrame(columns=['prob', 'cost'])

# Iterating through cut-off probability values between 0.1 and 0.5
idx = 0
for each_prob in range(10, 50):
    # Calculate total cost for each cut-off probability value
    cost = get_total_cost(y_pred_df.actual, y_pred_df.predicted_prob.map(lambda x: 1 if x > (each_prob / 100) else 0), 1, 5)
    cost_df.loc[idx] = [(each_prob / 100), cost]
    idx += 1

y_pred_df['predicted_using_cost'] = y_pred_df.predicted_prob.map(lambda x: 1 if x > 0.14 else 0)

# Drawing the confusion matrix for the new predictions based on cost analysis
draw_cm(y_pred_df.actual, y_pred_df.predicted_using_cost)
# ======================================================================================================================================================
bank_df = pd.read_csv('C:\\Users\\My PC\\Desktop\\Internship\\Machine Learning (Codes and Data Files)\\Data\\bank.csv')
X_features = list(bank_df.columns)
X_features.remove('subscribed')
encoded_df = pd.get_dummies(bank_df[X_features],drop_first=True)

Y = bank_df['subscribed'].map( lambda x : int(x == 'yes') )
X = encoded_df
bank_model1 = sm.Logit(Y,sm.add_constant(X).astype(int)).fit()

def get_significant_vars(lm):
    var_p_vals_df = pd.DataFrame(lm.pvalues)
    var_p_vals_df['vars'] = var_p_vals_df.index
    var_p_vals_df.columns = ['pvals', 'vars']
    # Returning the list of variables with p-value <= 0.05
    return list(var_p_vals_df[var_p_vals_df.pvals <= 0.05]['vars'])
significant_vars = get_significant_vars(bank_model1)
# print(significant_vars)
X_features = significant_vars[:]
X_features.remove('const')
bank_model2 = sm.Logit( Y, sm.add_constant(X[X_features].astype(int))).fit()
print(bank_model2.summary2())

pred_y_df = pd.DataFrame({
    'actual' : Y,
    'predicted_prob' : bank_model2.predict(sm.add_constant(X[X_features].astype(int)))
})

# sort predictions
sorted_pred_y_df = pred_y_df[[
    'predicted_prob','actual'
]].sort_values('predicted_prob',ascending=False)

# segment all observations into deciles
num_per_decile = int(len(sorted_pred_y_df)/10)
# print('numbers of observations per decile,',num_per_decile)

def get_deciles(df):
    # Initialize the 'decile' column with 1
    df['decile'] = 1
    idx = 0

    # Iterate through all 10 deciles
    for each_d in range(0, 10):
        # Assign each set of observations to a decile
        df.iloc[
            idx:idx + num_per_decile, df.columns.get_loc('decile')
        ] = each_d
        idx += num_per_decile

    # Increment decile values by 1 to make them 1-based instead of 0-based
    df['decile'] = df['decile'] + 1
    return df

deciles_pred_y_df = get_deciles(sorted_pred_y_df)
print(deciles_pred_y_df[:10])

gain_lift_df = pd.DataFrame(
    deciles_pred_y_df.groupby('decile')['actual'].sum()
).reset_index()
gain_lift_df.columns =['deciles','gain']
gain_lift_df['gain_percentage']=(100 * gain_lift_df.gain.cumsum()/gain_lift_df.gain.sum())
# print(gain_lift_df)

# plt.figure(figsize=(8,4))
# plt.plot(gain_lift_df['deciles'], gain_lift_df['gain_percentage'],'-')
# plt.show()

# calculate lift
gain_lift_df['lift'] = (gain_lift_df.gain_percentage)/(gain_lift_df.deciles * 10)
plt.figure(figsize=(8,4))
plt.plot(gain_lift_df['deciles'], gain_lift_df['lift'],'-')
# plt.show()


# split data
Y = credit_df.status
X = encoded_credit_df
train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.3,random_state=42)

"""
Gini vs. Entropy?   
Gini is faster & easier to compute.
Entropy gives more detailed splits but takes longer.
Most cases → Gini is preferred unless you want fine-grained control.
"""
"""
    Controls how big the tree can grow (helps prevent overfitting).
    max_depth → Limits tree depth (e.g., max_depth=3 stops at 3 levels).
    min_samples_split → Minimum samples required to split a node.
    min_samples_split=2 (default) means a node splits if it has at least 2 samples.
    min_samples_leaf → Minimum samples per leaf node.
    min_samples_leaf=1 (default) means leaves can have just 1 sample.
    Higher values prevent small branches.
    Tip:Use max_depth and min_samples_leaf to prevent overfitting.
"""
clf_tree = DecisionTreeClassifier(
    criterion='gini',max_depth=3
)

clf_tree.fit(train_x,train_y)

tree_predict = clf_tree.predict(test_x)
print(metrics.roc_auc_score(test_y,tree_predict))

export_graphviz(
    clf_tree,
    out_file='child_tree.dot',
    feature_names=train_x.columns,
    filled= True
)

# read created file
chd_tree_graph = Source.from_file('child_tree.dot')
chd_tree_graph.render('child_tree', format='png', cleanup=True)
# render png file
Image(filename='child_tree.png')

gini_node_1 = 1 - pow(491/700,2) - pow(209/700,2)
print('gini score: ',round(gini_node_1,4))

# Entropy
"""
Entropy measures uncertainty (randomness) in data. 📊
👉 If a node has only one class, entropy = 0 (pure).
👉 If a node has a 50-50 split, entropy = 1 (max uncertainty).

✅ Entropy closer to 0 → GOOD (Pure, Clear Decision)
❌ Entropy closer to 1 → BAD (Uncertain, Hard to Split)
"""

clf_tree_entropy = DecisionTreeClassifier(
    criterion='entropy',
    max_depth = 3
)
clf_tree_entropy.fit(train_x,train_y)

export_graphviz(
    clf_tree_entropy,
    out_file = 'chd_tree_entropy.dot',
    feature_names=train_x.columns
)
chd_tree_graph = pdot.graphviz.graph_from_dot_file('chd_tree_entropy.dot')
chd_tree_graph.write_jpg('chd_tree_entropy.png')
Image('chd_tree_entropy.png')

entropy_node_1 =  - (491/700)*math.log2(491/700) - (209/700)*math.log2(209/700)
print('Entropy: ',entropy_node_1)

tree_predict = clf_tree_entropy.predict(test_x)
print('AUC: ',metrics.roc_auc_score(test_y,tree_predict))

turned_parameters = [{
    'criterion' : ['gini','entropy'],
    'max_depth' : range(2,10)
}]

clf_tree = DecisionTreeClassifier()
clf = GridSearchCV(
    clf_tree,
    turned_parameters,
    cv = 10,
    scoring = 'roc_auc'
)
clf.fit(train_x,train_y)
print(clf.best_score_)
print(clf.best_params_)