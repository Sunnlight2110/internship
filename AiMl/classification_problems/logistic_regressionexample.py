import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

hr_df = pd.read_csv('C:\\Users\\My PC\\Desktop\\Internship\\Machine Learning (Codes and Data Files)\\Data\\hr_data.csv')
# print(hr_df.info())

hr_df['Status'] = hr_df['Status'].map(lambda x: int(x == 'Not Joined'))
# print(hr_df.to_string())
# print(hr_df.Status.value_counts())

X_features = list(hr_df.columns)
X_features.remove('Status')
# print(X_features)

encoded_df = pd.get_dummies(hr_df[X_features],drop_first=True).astype(int)
# print(list(encoded_df.columns))

X = sm.add_constant(encoded_df)
Y = hr_df.Status


vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
high_vif_features = vif_data[vif_data["VIF"] > 10]["Feature"].tolist()
print(high_vif_features)
X = X.drop(columns=high_vif_features)

vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
high_vif_features = vif_data[vif_data["VIF"] > 10]["Feature"].tolist()
print(high_vif_features)
X = X.drop(columns=high_vif_features)
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
high_vif_features = vif_data[vif_data["VIF"] > 10]["Feature"].tolist()
print(high_vif_features)

train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.3,random_state=42)

model1 = LogisticRegression(solver='liblinear', C=1.0)
model1.fit(train_x,train_y)

pred_y = model1.predict(test_x)
# Accuracy
accuracy = accuracy_score(test_y, pred_y)

# Confusion Matrix
cm = confusion_matrix(test_y, pred_y)

# Precision, Recall, F1-Score
precision = precision_score(test_y, pred_y)
recall = recall_score(test_y, pred_y)
f1 = f1_score(test_y, pred_y)

# print(f"Accuracy: {accuracy}")
# print(f"Confusion Matrix:\n{cm}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1-Score: {f1}")

# Undersampling
majority_class = hr_df[hr_df['Status'] == 0]
minority_class = hr_df[hr_df['Status'] == 1]

majority_class_undersampled = resample(
    majority_class,
    replace = False,
    n_samples= len(minority_class),
    random_state= 42
)

hr_df_balanced = pd.concat([majority_class_undersampled,minority_class])
# print(hr_df_balanced.value_counts())


encoded_df_balanced = pd.get_dummies(hr_df_balanced.drop('Status', axis=1)).astype(int)
X_balanced = sm.add_constant(encoded_df_balanced)
Y_balanced = hr_df_balanced['Status']

# print('x',X_balanced.shape)
# print('y',Y_balanced.shape)
train_x,test_x,train_y,test_y = train_test_split(X_balanced,Y_balanced,test_size=0.3,random_state=42)



model2 = LogisticRegression(solver='liblinear', C=1.0)
model2.fit(train_x,train_y)

pred_y = model2.predict(test_x)
# Accuracy
accuracy = accuracy_score(test_y, pred_y)

# Confusion Matrix
cm = confusion_matrix(test_y, pred_y)

# Precision, Recall, F1-Score
precision = precision_score(test_y, pred_y)
recall = recall_score(test_y, pred_y)
f1 = f1_score(test_y, pred_y)

# print(f"Accuracy: {accuracy}")
# print(f"Confusion Matrix:\n{cm}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1-Score: {f1}")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(encoded_df_balanced)
pca = PCA(n_components=0.8)
X_pca = pca.fit_transform(X_scaled)

explain_variance = pca.explained_variance_ratio_
print(f"Top 10 components explain {sum(explain_variance) * 100:.2f}% of the variance")

train_x, test_x, train_y, test_y = train_test_split(X_pca, Y_balanced, test_size=0.3, random_state=42)
hr_model3 = LogisticRegression(solver='liblinear', C=1.0)
hr_model3.fit(train_x, train_y)

pred_y = hr_model3.predict(test_x)
accuracy = accuracy_score(test_y, pred_y)

# Confusion Matrix
cm = confusion_matrix(test_y, pred_y)

# Precision, Recall, F1-Score
precision = precision_score(test_y, pred_y)
recall = recall_score(test_y, pred_y)
f1 = f1_score(test_y, pred_y)

# print(f"Accuracy: {accuracy}")
# print(f"Confusion Matrix:\n{cm}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1-Score: {f1}")

X_balanced = encoded_df_balanced
Y_balanced = hr_df_balanced['Status']

train_x,test_x,train_y,test_y = train_test_split(X_balanced,Y_balanced,test_size=0.3,random_state=42)
clf_tree = DecisionTreeClassifier(criterion='gini',max_depth=4)
clf_tree.fit(train_x,train_y)
tree_predict = clf_tree.predict(test_x)
print("actual roc_auc_score",metrics.roc_auc_score(test_y,tree_predict))
# export_graphviz(
#     clf_tree,
#     out_file = 'hr_rules.dot',
#     feature_names=train_x.columns,
#      class_names=["Not Joining", "Joining"],  # Adjust class names based on your dataset
#     filled=True,  # Enables coloring
#     rounded=True,  # Rounds the corners
#     special_characters=True
# )
# chd_tree_graph = pdot.graphviz.graph_from_dot_file('hr_rules.dot')
# chd_tree_graph.write_jpg('hr_rules.png')
# Image('hr_rules.png')

import numpy as np

# Access the tree structure
n_nodes = clf_tree.tree_.node_count  # Total nodes
children_left = clf_tree.tree_.children_left  # Left child of each node
children_right = clf_tree.tree_.children_right  # Right child of each node
feature = clf_tree.tree_.feature  # Feature used for splitting
threshold = clf_tree.tree_.threshold  # Threshold value for the split
values = clf_tree.tree_.value  # Class distribution at each node
gini = clf_tree.tree_.impurity  # Gini impurity at each node

# Loop through each node and print details
for node in range(n_nodes):
    # Extract class distribution
    class_counts = values[node][0]  # Values are stored as an array inside an array
    total_samples = np.sum(class_counts)  # Total samples at node

    # Manual Gini Calculation
    gini_manual = 1 - np.sum((class_counts / total_samples) ** 2) if total_samples > 0 else 0

    print(f"Node {node}:")
    print(f" - Feature Used: {train_x.columns[feature[node]] if feature[node] != -2 else 'Leaf Node'}")
    print(f" - Threshold: {threshold[node] if feature[node] != -2 else 'N/A'}")
    print(f" - Gini (sklearn): {gini[node]}")
    print(f" - Gini (manual calculation): {round(gini_manual, 4)}")
    print(f" - Values: {class_counts}")
    print(f" - Left Child: {children_left[node]}")
    print(f" - Right Child: {children_right[node]}\n")

"""Relocation Matters (Node 0)
Problem: If a candidate isn’t open to relocation, rejection is HIGH.
HR Fix: Offer relocation assistance or remote work options for non-relocating candidates!
2️⃣ Notice Period (Node 2)
Problem: If a candidate’s notice period is above 37.5 days, rejection chances increase.
HR Fix:
Fast-track hiring for candidates with a high notice period.
Offer early joining bonuses to encourage quick transitions.
3️⃣ Accepting Offer Delay (Node 3 & 11)
Problem: Candidates who take too long to accept an offer (>23.5 days) often drop out.
HR Fix:
Reduce offer processing time to avoid candidate hesitation.
Keep follow-ups active within 24-48 hours after making an offer.
Provide immediate clarity on salary, benefits, and growth to speed up acceptance.
4️⃣ Salary & Compensation (Node 10)
Problem: Candidates with a low CTC difference (i.e., not getting a big salary jump) are rejecting offers.
HR Fix:
Ensure competitive salary hikes.
Offer performance bonuses, stock options, or benefits instead of just base salary.
5️⃣ Candidate Source (Node 4)
Problem: Candidates from recruitment agencies have a higher rejection rate than direct applicants.
HR Fix:
Reduce dependency on agencies and focus on internal referrals or direct job postings.
Build a stronger employer brand so people apply directly instead of relying on agencies."""


# turned_parameters = [{
#     'criterion' : ['gini','entropy'],
#     'max_depth' : range(2,10),
#     'min_samples_split': [2, 5, 10, 20],
#     'min_samples_leaf': [1, 5, 10, 20]
# }]

# strat_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# clf_tree = DecisionTreeClassifier()
# clf = GridSearchCV(
#     clf_tree,
#     turned_parameters,
#     cv = strat_kfold,
#     scoring = 'roc_auc'
# )
# clf.fit(train_x,train_y)
# print("best_score",clf.best_score_)
# print("best_params",clf.best_params_)

# train_pred = clf.predict(train_x)
# print("Train ROC AUC:", metrics.roc_auc_score(train_y, train_pred))
# print("Test ROC AUC:", metrics.roc_auc_score(test_y, tree_predict))

