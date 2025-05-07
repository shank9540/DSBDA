# Step 1: Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load the Titanic dataset
titanic = sns.load_dataset('titanic')

# Step 3: Plot the box plot for 'age' with respect to 'sex' and survival status ('survived')
plt.figure(figsize=(10, 6))
sns.boxplot(x='sex', y='age', hue='survived', data=titanic, palette="Set2")

# Adding labels and title
plt.title('Distribution of Age by Gender and Survival Status')
plt.xlabel('Gender')
plt.ylabel('Age')

# Display the plot
plt.show()
