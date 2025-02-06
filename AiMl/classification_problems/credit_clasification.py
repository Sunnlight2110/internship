import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import metrics
# %matplotlib inline

credit_df = pd.read_csv('C:\\Users\\My PC\\Desktop\\Internship\\Machine Learning (Codes and Data Files)\\Data\\German Credit Data.csv')
print(credit_df.info())
# print(credit_df.iloc[0:5,1:7])
# print(credit_df.iloc[0:5,7:])

# print(credit_df.status.value_counts())  #? gives status column which will be used as dependent variable
# credit_df.dropna()

X_features = list(credit_df.columns)
X_features.remove('status')
# print(X_features)

# # encoding
encoded_df = pd.get_dummies(credit_df[X_features], drop_first=True).astype(int)
# print(list(encoded_df.columns))
# print(encoded_df[
#     ['checkin_acc_A12',
#     'checkin_acc_A13',
#     'checkin_acc_A14',]
# ].head(5))

# ?add constants
Y = credit_df.status    #Independent variable
X = sm.add_constant(encoded_df) #Dependent variable


train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.3,random_state=42)

# # ? check for multi-collinearity and remove them
# # Step 1: Identify highly correlated features
# corr_matrix = train_x.corr()
# corr_matrix_no_diag = corr_matrix.where(~np.eye(corr_matrix.shape[0],dtype=bool))
# high_corr = corr_matrix_no_diag[abs(corr_matrix_no_diag) > 0.9]

# # Step 2: Drop highly correlated features
# cols_to_drop = set()
# for feature in high_corr.columns:
#     if any(high_corr[feature].abs() > 0.9):
#         cols_to_drop.add(feature)

# train_x_filtered = train_x.drop(columns=cols_to_drop)

# Step 3: Check for constant columns
# train_x_filtered = train_x.loc[:, (train_x != train_x.iloc[0]).any()]

# # Step 4: (Optional) Standardize the features
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# train_x_scaled = scaler.fit_transform(train_x_filtered)     #Transoms features to have mean = 0 and std = 1, it equals all weight

# # Step 5: Fit the Logistic Regression model
credit_df_model1 = sm.Logit(train_y, train_x).fit(maxiter=2000)
print(credit_df_model1.summary2())

def get_significant_vars(lm):
    # stores p-values and corresponding column names in a dataframe
    var_p_values_df = pd.DataFrame(lm.summary2().tables[1]['P>|z|'])  # Stores P value
    var_p_values_df['vars'] = var_p_values_df.index  # stores variable names
    var_p_values_df.columns = ['pvals', 'vars']  # Rename columns
    # Print column names and p-values for debugging
    print("Variables and their p-values:")
    print(var_p_values_df)
    return list(var_p_values_df[var_p_values_df.pvals <= 0.05]['vars'])

significant_vars = get_significant_vars(credit_df_model1)
# print("Significant variables:", significant_vars)

# # Make sure we're using variables that exist in train_x
# valid_vars = [var for var in significant_vars if var in train_x.columns]
credit_df_model2 = sm.Logit(train_y, sm.add_constant(train_x[significant_vars])).fit()
# print(credit_df_model2.summary2())  

y_pred_df = pd.DataFrame({
    'actual': test_y,
    'predicted_prob': credit_df_model2.predict(
        sm.add_constant(test_x[significant_vars])
    )
})

# print(y_pred_df.sample(10))

y_pred_df['predicted'] = y_pred_df.predicted_prob.map(lambda X: 1 if X > 0.5 else 0)
# print(y_pred_df.to_string())

def draw_cm(actual,predicted):
    """
    This function takes the actual and predicted labels from a classification model 
    and generates a confusion matrix plot to visualize model performance.

    Args:
    actual (array-like): True labels of the dataset (ground truth).
    predicted (array-like): Predicted labels from the model.

    Returns:
    None: This function directly displays a heatmap of the confusion matrix.
    
    The confusion matrix provides insights into:
        - True Positives (TP): Correctly predicted positive cases.
        - True Negatives (TN): Correctly predicted negative cases.
        - False Positives (FP): Incorrectly predicted positive cases.
        - False Negatives (FN): Incorrectly predicted negative cases.

    The heatmap visually represents these values with a color gradient, 
    making it easy to spot where the model is performing well or making mistakes.
    """
    cm = metrics.confusion_matrix(actual,predicted, labels=[1,0])
    sn.heatmap(
        cm, annot= True, fmt='2f',
        xticklabels=['Bed credit','Good credit'],
        yticklabels=['Bed credit','Good credit']
    )
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# draw_cm(y_pred_df.actual,y_pred_df.predicted)

# print(metrics.classification_report(
#     y_pred=y_pred_df.predicted,
#     y_true=y_pred_df.actual
# ))

# plt.figure(figsize=(8,6))
# # plotting distribution of predicted probability value for bed credits
# sn.histplot(
#     y_pred_df[y_pred_df.actual == 1]['predicted_prob'],
#     bins=30, color='b', label='bad credit', kde=False
# )

# # Plotting distribution of predicted probability for good credit
# sn.histplot(
#     y_pred_df[y_pred_df.actual == 0]['predicted_prob'],
#     bins=30, color='g', label='good credit', kde=False
# )

# plt.legend()
# plt.show()


# Plot ROC curve
def draw_roc(actual, probs):
    """
    This function plots the Receiver Operating Characteristic (ROC) curve and calculates the 
    Area Under the Curve (AUC) to evaluate the performance of a classification model.

    Args:
    actual (array-like): True labels (0 or 1) of the test dataset.
    probs (array-like): Predicted probabilities for the positive class (values between 0 and 1).

    Returns:
    fpr (array): False positive rates at different thresholds.
    tpr (array): True positive rates at different thresholds.
    thresholds (array): Threshold values used to compute the TPR and FPR.
    auc_value (float): Area under the ROC curve (AUC), a measure of the model's ability to distinguish 
                        between the positive and negative classes.

    The ROC curve shows the trade-off between the True Positive Rate (TPR) and False Positive Rate (FPR) 
    as the decision threshold is varied. The AUC score quantifies how well the model separates the classes, 
    with 1 being perfect and 0.5 representing random guessing.

    The function also displays a plot with:
    - The ROC curve showing the TPR vs. FPR.
    - A random guess line (diagonal) for reference.
    - A legend indicating the ROC curve and AUC value.

    Example:
    fpr, tpr, thresholds, auc_value = draw_roc(actual_labels, predicted_probs)
    print(f"AUC: {auc_value}")
    """
    fpr, tpr, thresholds = metrics.roc_curve(actual, probs)
    auc_value = metrics.roc_auc_score(actual, probs)  # Calculate AUC
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_value:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='best')
    # plt.show()
    return fpr, tpr, thresholds, auc_value

# # Draw ROC and get values
# fpr, tpr, thresholds, auc_value = draw_roc(y_pred_df.actual, y_pred_df.predicted_prob)
# print(f"AUC: {auc_value}")

# tpr_fpr = pd.DataFrame({
#     'tpr':tpr,
#     'fpr':fpr,
#     'thresholds':thresholds,
# })
# tpr_fpr['diff'] = tpr_fpr.fpr - tpr_fpr.tpr
# print(tpr_fpr.sort_values('diff',ascending=False)[0:5])

# y_pred_df['predicted_new'] = y_pred_df.predicted_prob.map(lambda x: 1 if x > 0.2 else 0)
# # y_pred_df['predicted_new'] = y_pred_df.predicted_prob.map(lambda X: 1 if X > 0.5 else 0)
# draw_cm(y_pred_df.actual,y_pred_df.predicted_new)
# print(metrics.classification_report(y_pred_df.actual,y_pred_df.predicted_new))


def get_total_cost(actual, predicted, cost_FPs, costFNs):
    cm = metrics.confusion_matrix(actual, predicted, labels=[1, 0])
    cm_mat = np.array(cm)
    return (cm_mat[0, 1] * costFNs) + (cm_mat[1, 0] * cost_FPs)

cost_df = pd.DataFrame(columns=['prob', 'cost'])
idx = 0
for each_prob in range(10, 50):
    cost = get_total_cost(
        y_pred_df.actual,
        y_pred_df.predicted_prob.map(lambda x: 1 if x > (each_prob / 100) else 0),
        1, 5
    )
    cost_df.loc[idx] = [(each_prob / 100), cost]
    idx += 1

print(cost_df.sort_values('cost', ascending=True)[:5])

y_pred_df['predicted_using_cost'] = y_pred_df.predicted_prob.map(lambda x: 1 if x > 0.14 else 0)
# draw_cm(
#     y_pred_df.actual,
#     y_pred_df.predicted_using_cost
# )
print(metrics.classification_report(y_pred_df.actual,y_pred_df.predicted_using_cost))