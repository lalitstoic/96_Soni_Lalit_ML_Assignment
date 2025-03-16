import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Correct file path
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
df_no_outliers = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

print("Shape before:", df.shape)
print("Shape after removing outliers:", df_no_outliers.shape)


#Normalization of the data

df = pd.read_csv("./heart.csv")

scaler = MinMaxScaler()  # Creates a scaler
df[['chol', 'trestbps']] = scaler.fit_transform(df[['chol', 'trestbps']])

print(df.head())

#standarization of the data

scaler = StandardScaler()
df[['thalach', 'trestbps']] = scaler.fit_transform(df[['thalach', 'trestbps']])

print(df.head())

