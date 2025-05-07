# Step 1: Import required libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load the Titanic dataset from Seaborn's inbuilt datasets
titanic = sns.load_dataset('titanic')

# Step 3: Display first few rows of the dataset to understand its structure
print(titanic.head())

# Step 4: Explore the data using seaborn's pairplot to find patterns
# A pairplot will display pairwise relationships between columns.
sns.pairplot(titanic.dropna())  # Dropping missing values for this plot
plt.suptitle("Pairplot of Titanic Dataset", y=1.02)
plt.show()

# Step 5: Plot a histogram of the 'fare' column to check the distribution of ticket prices
sns.histplot(titanic['fare'], kde=True, bins=30)  # Adding kde=True will plot the Kernel Density Estimate (KDE)
plt.title("Distribution of Ticket Prices (Fare)")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.show()
