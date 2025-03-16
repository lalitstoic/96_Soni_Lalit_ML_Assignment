import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

import pickle

df = pd.read_csv('./heart.csv')

# Displaying the missing values
print("Missing Values in the Data set : ")
print(df.isnull().sum())  # Shows the count of missing values per column

# Filling missing values with the mean of each column
df.fillna(df.mean(), inplace=True)

# Load the dataset and display the first few rows.
print("Data set after filling the missing values : ")
print(df.head())

# Performing exploratory data analysis ( EDA )
# Get summary statistics
print("Exploratory Data Analysis : ")
print("Summary of numeric columns : ")
print(df.describe())  # Summary of numeric columns
print("Data types and Missing values : ")
print(df.info())  # Data types & missing values

# Histogram for all numeric columns
df.hist(figsize=(10, 8), bins=20)
# plt.show()

# Box plot for Age levels
sns.boxplot(x=df['age'])
# plt.show()
sns.boxplot(x=df['sex'])
# plt.show()
sns.boxplot(x=df['cp'])
# plt.show()
sns.boxplot(x=df['trestbps'])
# plt.show()
sns.boxplot(x=df['chol'])
# plt.show()
sns.boxplot(x=df['fbs'])
# plt.show()
sns.boxplot(x=df['restecg'])
# plt.show()
sns.boxplot(x=df['thalach'])
# plt.show()
sns.boxplot(x=df['exang'])
# plt.show()
sns.boxplot(x=df['oldpeak'])
# plt.show()
sns.boxplot(x=df['slope'])
# plt.show()
sns.boxplot(x=df['ca'])
# plt.show()
sns.boxplot(x=df['thal'])
# plt.show()
sns.boxplot(x=df['target'])
# plt.show()

# Compute correlation matrix
corr_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
# plt.show()

Q1 = df.quantile(0.25)  # 25th percentile
Q3 = df.quantile(0.75)  # 75th percentile
IQR = Q3 - Q1  # Interquartile range

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
print("After removing outlier : ")
df_no_outliers = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

print("Shape before:", df.shape)
print("Shape after removing outliers:", df_no_outliers.shape)


#Normalization of the data

print("Normalization of the Data : ")
df = pd.read_csv("./heart.csv")

scaler = MinMaxScaler()  # Creates a scaler
df[['chol', 'trestbps']] = scaler.fit_transform(df[['chol', 'trestbps']])

print(df.head())

#standarization of the data

print("Standarization of the Data : ")
scaler = StandardScaler()
df[['thalach', 'trestbps']] = scaler.fit_transform(df[['thalach', 'trestbps']])

print(df.head())


#1.	Handle categorical variables (one-hot encoding, label encoding).

# One-Hot Encoding for non-ordinal categorical features
df = pd.get_dummies(df, columns=['cp'], drop_first=True)  # Drops one column to avoid multicollinearity

# Label Encoding for ordinal categories 
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])  # Converts 'Male'/'Female' to 0/1

print("Hot Encoding : ")
print(df.head())


# Identify and remove highly correlated features.

# Compute correlation matrix
corr_matrix = df.corr()

# Visualizing correlations
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
# plt.show()

# Remove features with correlation > 0.85
high_corr_features = set()
threshold = 0.85  # Define correlation threshold

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            colname = corr_matrix.columns[i]
            high_corr_features.add(colname)

df.drop(high_corr_features, axis=1, inplace=True)
print("Remaining features:", df.columns)

#Apply feature selection techniques (e.g., SelectKBest, Mutual Information).


# Assuming the target column is the last one
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target

# 🔹 Ensure all values are non-negative for chi2
X_non_negative = X.copy()
X_non_negative = X_non_negative.abs()

# 🔹 Normalize the data for better performance
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 🔹 Feature Selection using SelectKBest (ANOVA F-test)
selector_f = SelectKBest(score_func=f_classif, k=5)  # Select top 5 features
X_new_f = selector_f.fit_transform(X_scaled, y)
selected_features_f = X.columns[selector_f.get_support()]
print("Top features (ANOVA F-test):", list(selected_features_f))

# 🔹 Feature Selection using Mutual Information
selector_mi = SelectKBest(score_func=mutual_info_classif, k=5)  # Select top 5 features
X_new_mi = selector_mi.fit_transform(X_scaled, y)
selected_features_mi = X.columns[selector_mi.get_support()]
print("Top features (Mutual Information):", list(selected_features_mi))



#Part 3: Model Development & Training (35 Marks)

# Splitting the dataset (80-20 split)
X = df.drop(columns=['target'])  # Features
y = df['target']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardizing the Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression Model
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)

# Train Random Forest Model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Hyperparameter Tuning for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_  # Best Random Forest model

# Evaluation Function
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"--- {model_name} ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Model Evaluation
evaluate_model(logreg, X_test, y_test, "Logistic Regression")
evaluate_model(best_rf, X_test, y_test, "Best Random Forest")



#part 4




# 1️ Generate Dummy Data (Replace with your real dataset)
X, y = make_classification(n_samples=205, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2️ Train Logistic Regression & Random Forest
log_reg = LogisticRegression()
rf = RandomForestClassifier(n_estimators=100, random_state=42)

log_reg.fit(X_train, y_train)
rf.fit(X_train, y_train)

# 3️ Predictions & Confusion Matrices
y_pred_log = log_reg.predict(X_test)
y_pred_rf = rf.predict(X_test)

cm_log = confusion_matrix(y_test, y_pred_log)
cm_rf = confusion_matrix(y_test, y_pred_rf)

# 4️ Plot Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm_log, annot=True, fmt='d', cmap="Blues", ax=axes[0])
axes[0].set_title("Logistic Regression Confusion Matrix")

sns.heatmap(cm_rf, annot=True, fmt='d', cmap="Greens", ax=axes[1])
axes[1].set_title("Random Forest Confusion Matrix")
plt.show()

# 5️ Precision-Recall Curve
y_scores_log = log_reg.decision_function(X_test)
y_scores_rf = rf.predict_proba(X_test)[:, 1]

precision_log, recall_log, _ = precision_recall_curve(y_test, y_scores_log)
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_scores_rf)

plt.plot(recall_log, precision_log, label=f"Logistic Regression (AUC={auc(recall_log, precision_log):.2f})", color='blue')
plt.plot(recall_rf, precision_rf, label=f"Random Forest (AUC={auc(recall_rf, precision_rf):.2f})", color='green')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# 6️ Feature Selection
selector = SelectKBest(score_func=f_classif, k=5)  # Selecting top 5 features
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Retrain with Selected Features
log_reg.fit(X_train_selected, y_train)
rf.fit(X_train_selected, y_train)

# New Predictions
y_pred_log_sel = log_reg.predict(X_test_selected)
y_pred_rf_sel = rf.predict(X_test_selected)

# Print Reports
print("--- Logistic Regression After Feature Selection ---")
print(classification_report(y_test, y_pred_log_sel))

print("--- Random Forest After Feature Selection ---")
print(classification_report(y_test, y_pred_rf_sel))









# Assuming 'best_model' is your best-performing trained model (e.g., Random Forest)
best_model = best_rf  # Replace with the actual variable of your trained model

# Save the model to a file
with open("model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("Model saved successfully as model.pkl")