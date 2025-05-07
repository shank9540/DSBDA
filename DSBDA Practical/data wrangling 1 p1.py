# Step 1: Import required libraries
import pandas as pd
import numpy as np

# Step 2 & 3: Load the Iris dataset from an online source
url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
df = pd.read_csv(url)

# Step 4: Data Preprocessing
print("First 5 rows of the dataset:")
print(df.head())

print("\nSummary Statistics:")
print(df.describe())

print("\nChecking for missing values:")
print(df.isnull().sum())

print("\nShape of the DataFrame (rows, columns):")
print(df.shape)

print("\nVariable Descriptions:")
print(df.columns)

# Step 5: Data Formatting and Normalization
print("\nData types of each column before conversion:")
print(df.dtypes)

# Convert all column names to lowercase for consistency
df.columns = df.columns.str.lower()

# Ensure numerical columns are float
df['sepal_length'] = df['sepal_length'].astype(float)
df['sepal_width'] = df['sepal_width'].astype(float)
df['petal_length'] = df['petal_length'].astype(float)
df['petal_width'] = df['petal_width'].astype(float)

print("\nData types after conversion (if any):")
print(df.dtypes)

# Step 6: Convert categorical variables to numeric
# Use Label Encoding for the 'species' column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['species_encoded'] = le.fit_transform(df['species'])

print("\nUnique classes in 'species' column:")
print(df['species'].unique())

print("\nData after encoding 'species' column:")
print(df[['species', 'species_encoded']].head())

# Final DataFrame overview
print("\nFinal DataFrame with encoded column:")
print(df.head())
