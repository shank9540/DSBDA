# Step 1: Import required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load the Iris dataset from Seaborn's inbuilt datasets (or from an external URL)
# In this case, we'll use the URL from the provided link (Iris dataset is available in seaborn too).
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv(url, names=columns)

# Step 3: Display the first few rows to understand the dataset
print(df.head())

# Step 4: List features and their types
print("\nFeatures and their types:")
feature_types = df.dtypes
print(feature_types)

# Step 5: Create histograms for each feature
df.dropna(inplace=True)  # Drop any NaN values before plotting
plt.figure(figsize=(12, 10))

# Plotting histograms
for i, feature in enumerate(df.columns[:-1]):  # Exclude 'species' from numeric features
    plt.subplot(2, 2, i+1)
    sns.histplot(df[feature], kde=True, bins=20, color='blue')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Step 6: Create box plots for each feature
plt.figure(figsize=(12, 10))

# Plotting boxplots
for i, feature in enumerate(df.columns[:-1]):  # Exclude 'species' from numeric features
    plt.subplot(2, 2, i+1)
    sns.boxplot(x=df[feature], color='orange')
    plt.title(f'Boxplot of {feature}')
    plt.xlabel(feature)

plt.tight_layout()
plt.show()
