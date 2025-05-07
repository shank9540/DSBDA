# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load the dataset
df = pd.read_csv('C:\DSBDA Practical\Data Analytics 2 p8/Social_Network_Ads.csv')

# Step 3: Exploratory Data Analysis (EDA) - Optional
print(df.head())  # Check the first few rows of the dataset

# Step 4: Preprocess the data
# Select relevant columns and drop any irrelevant ones if needed
X = df[['Age', 'EstimatedSalary']]  # Features: Age and Estimated Salary
y = df['Purchased']  # Target: Purchased (1 or 0)

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Feature Scaling (Logistic Regression performs better with normalized data)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Initialize and train the Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Step 8: Make predictions on the test set
y_pred = log_reg.predict(X_test)

# Step 9: Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')

# Step 10: Visualize the results - Optional (Visualizing the decision boundary for training data)
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', s=50, edgecolors='k')
plt.title('Logistic Regression (Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.show()

# Visualize on the Test Set
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', s=50, edgecolors='k')
plt.title('Logistic Regression (Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.show()
