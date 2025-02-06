# Importing necessary libraries and suppressing warnings
import warnings 
warnings.filterwarnings('ignore') 

import pandas as pd 
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import seaborn as sn 
from sklearn import metrics

# Reading the dataset
credit_df = pd.read_csv('C:\\Users\\My PC\\Desktop\\Internship\\Machine Learning (Codes and Data Files)\\Data\\German Credit Data.csv')

# Extracting feature columns (excluding 'status' which is the target variable)
X_features = list(credit_df.columns) 
X_features.remove('status')

# Encoding categorical features into dummy/indicator variables for machine learning
# pd.get_dummies() is used to convert categorical variables into numerical values
encoded_credit_df = pd.get_dummies(credit_df[X_features], drop_first=True).astype(int)

# Setting the target variable 'status' (0 for good credit, 1 for bad credit)
Y = credit_df.status 

# Adding a constant (intercept) column to X using sm.add_constant
X = sm.add_constant(encoded_credit_df)

# Splitting the data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Fitting a Logistic Regression model using Statsmodels' Logit function
logit = sm.Logit(y_train, X_train)
logit_model = logit.fit()

# Function to retrieve significant variables based on p-values less than or equal to 0.05
def get_significant_vars(lm):
    var_p_vals_df = pd.DataFrame(lm.pvalues)
    var_p_vals_df['vars'] = var_p_vals_df.index
    var_p_vals_df.columns = ['pvals', 'vars']
    # Returning the list of variables with p-value <= 0.05
    return list(var_p_vals_df[var_p_vals_df.pvals <= 0.05]['vars'])

# Getting significant variables from the model
significant_vars = get_significant_vars(logit_model)

# Fitting the logistic regression model with only the significant variables
final_logit = sm.Logit(y_train, sm.add_constant(X_train[significant_vars])).fit()

# Predicting probabilities for the test data based on the final model
y_pred_df = pd.DataFrame({
    "actual": y_test,  # Actual target values
    "predicted_prob": final_logit.predict(sm.add_constant(X_test[significant_vars]))  # Predicted probabilities
})

# Creating a binary prediction (1 if prob > 0.5 else 0)
y_pred_df['predicted'] = y_pred_df.predicted_prob.map(lambda x: 1 if x > 0.5 else 0)

# Displaying a sample of predictions
print(y_pred_df.sample(10, random_state=42))

# Function to plot the confusion matrix
def draw_cm(actual, predicted):
    """
    This function takes in the actual and predicted labels, and displays a confusion matrix as a heatmap.
    """
    cm = metrics.confusion_matrix(actual, predicted, labels=[1, 0])
    sn.heatmap(cm, annot=True, fmt='.2f', xticklabels=["Bad credit", "Good Credit"], yticklabels=["Bad credit", "Good Credit"])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Uncomment the following line to draw the confusion matrix
# draw_cm(y_pred_df.actual, y_pred_df.predicted)

# Uncomment the following line to print the classification report
# print(metrics.classification_report(y_pred_df.actual, y_pred_df.predicted))

# Function to plot the ROC curve
def draw_roc(actual, probs):
    """
    This function calculates and plots the Receiver Operating Characteristic (ROC) curve, 
    and returns the false positive rate (fpr), true positive rate (tpr), and thresholds used.
    """
    fpr, tpr, thresholds = metrics.roc_curve(actual, probs, drop_intermediate=False)
    auc_score = metrics.roc_auc_score(actual, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
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

# Sorting the DataFrame by 'diff' in descending order to find the best threshold
print(tpr_fpr.sort_values('diff', ascending=False)[0:5])

# Applying a new threshold (0.22) for prediction and checking performance
y_pred_df['predicted_new'] = y_pred_df.predicted_prob.map(lambda x: 1 if x > 0.22 else 0)

# Drawing the confusion matrix for the new threshold
draw_cm(y_pred_df.actual, y_pred_df.predicted_new)

# Printing the classification report for the new threshold
print(metrics.classification_report(y_pred_df.actual, y_pred_df.predicted_new))

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

# Uncomment the following line to see the top 5 probability values with the lowest cost
# print(cost_df.sort_values('cost', ascending=True)[0:5])

# Apply the optimal threshold based on cost analysis
y_pred_df['predicted_using_cost'] = y_pred_df.predicted_prob.map(lambda x: 1 if x > 0.14 else 0)

# Drawing the confusion matrix for the new predictions based on cost analysis
draw_cm(y_pred_df.actual, y_pred_df.predicted_using_cost)