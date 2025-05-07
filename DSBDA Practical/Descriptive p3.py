# Step 1: Import required libraries
import pandas as pd

# Step 2: Load the Iris dataset
url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
df = pd.read_csv(url)

# Step 3: Group by categorical variable 'species' and calculate summary statistics for numeric columns
grouped_stats = df.groupby('species').agg({
    'sepal_length': ['mean', 'median', 'min', 'max', 'std'],
    'sepal_width':  ['mean', 'median', 'min', 'max', 'std'],
    'petal_length': ['mean', 'median', 'min', 'max', 'std'],
    'petal_width':  ['mean', 'median', 'min', 'max', 'std']
})

print("Grouped Summary Statistics by Species:")
print(grouped_stats)

# Step 4: Create list of numeric values (sepal_length) for each species
sepal_length_groups = df.groupby('species')['sepal_length'].apply(list)

print("\nList of Sepal Length values by species:")
print(sepal_length_groups)

# Step 5: Display descriptive statistics for each species
for species in df['species'].unique():
    print(f"\nDescriptive statistics for {species}:")
    print(df[df['species'] == species].describe())
