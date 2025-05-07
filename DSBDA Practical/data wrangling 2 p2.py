# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Step 2: Create a sample "Academic Performance" dataset
data = {
    'Student_ID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'Math_Score': [88, 92, 79, 94, 85, None, 77, 100, 56, 150],  # Note the None and an outlier (150)
    'Science_Score': [90, 85, 78, 92, 88, 84, 76, None, 60, 99],
    'Attendance_%': [95, 98, 92, 97, 91, 88, None, 100, 85, 102],  # 102% is an inconsistency
}

df = pd.DataFrame(data)
print("Original Data:")
print(df)

# Step 3: Handling Missing Values
print("\nMissing values per column:")
print(df.isnull().sum())

# Fill missing Math_Score and Science_Score with column mean
df['Math_Score'].fillna(df['Math_Score'].mean(), inplace=True)
df['Science_Score'].fillna(df['Science_Score'].mean(), inplace=True)
df['Attendance_%'].fillna(df['Attendance_%'].mean(), inplace=True)

# Fix Attendance% inconsistency (>100%)
df['Attendance_%'] = df['Attendance_%'].apply(lambda x: 100 if x > 100 else x)

print("\nData after handling missing values and fixing inconsistencies:")
print(df)

# Step 4: Outlier Detection and Handling
# Using Z-score method
z_scores = np.abs(stats.zscore(df[['Math_Score', 'Science_Score', 'Attendance_%']]))
outliers = (z_scores > 3)
print("\nOutliers Detected (Z > 3):")
print(outliers)

# Replace outliers in Math_Score with median (e.g., 150 is an outlier)
math_median = df['Math_Score'].median()
df.loc[outliers[:, 0], 'Math_Score'] = math_median

print("\nData after handling outliers in Math_Score:")
print(df)

# Step 5: Data Transformation
# Transform Math_Score using log to reduce skewness
# (Add 1 to avoid log(0) if needed)
df['Math_Score_Log'] = np.log(df['Math_Score'] + 1)

print("\nData with Transformed 'Math_Score_Log':")
print(df[['Math_Score', 'Math_Score_Log']])

# Plot before and after transformation
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['Math_Score'], kde=True)
plt.title('Original Math Score Distribution')

plt.subplot(1, 2, 2)
sns.histplot(df['Math_Score_Log'], kde=True)
plt.title('Log Transformed Math Score Distribution')
plt.tight_layout()
plt.show()
