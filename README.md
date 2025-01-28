# Titanic - Machine Learning from Disaster Dataset: Data Cleaning and Preprocessing

This project demonstrates robust data cleaning and preprocessing techniques using the Titanic-Machine Learning from Disaster dataset. The goal is to prepare the data for further analysis and modeling by handling missing values, engineering features, encoding categorical variables, and normalizing numerical features.

## Steps

### Step 1: Load the Dataset
First, load the dataset into your environment.
** Loading the dataset is the first step to make the data available for analysis and preprocessing. **

```python
import pandas as pd

# Load the dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
```

### Step 2: Understand the Data
Get a basic understanding of the dataset by exploring its structure and summary statistics.
** Understanding the data helps identify the types of variables, detect any anomalies, and get an overview of the dataset's structure and content. **

```python
# Display the first few rows of the dataset
print(train_data.head())

# Get summary statistics
print(train_data.describe())

# Get information about the dataset
print(train_data.info())
```

### Step 3: Handle Missing Values
Identify and handle missing values in the dataset.
** Handling missing values is crucial to ensure the dataset is complete and to prevent errors during analysis and modeling. **

```python
# Check for missing values
print(train_data.isnull().sum())

# Fill missing values for 'Age' with the median
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())

# Fill missing values for 'Embarked' with the mode
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

# Drop the 'Cabin' column due to a high number of missing values
if 'Cabin' in train_data.columns:
    train_data.drop(columns=['Cabin'], inplace=True)
```

### Step 4: Feature Engineering
Create new features or transform existing ones to improve the model's performance.
** Feature engineering helps create new variables that can provide better insights and improve the predictive power of the model. **

```python
# Create a new feature 'FamilySize'
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1

# Create a new feature 'IsAlone'
train_data['IsAlone'] = (train_data['FamilySize'] == 1).astype(int)

# Extract titles from names
train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Simplify titles
train_data['Title'] = train_data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train_data['Title'] = train_data['Title'].replace('Mlle', 'Miss')
train_data['Title'] = train_data['Title'].replace('Ms', 'Miss')
train_data['Title'] = train_data['Title'].replace('Mme', 'Mrs')
```

### Step 5: Encode Categorical Variables
Convert categorical variables into numerical values.
** Encoding categorical variables is necessary because most machine learning algorithms require numerical input. **

```python
# Encode 'Sex' column
train'] = train_data['Sex'].map({'male': 0, 'female': 1}).astype(int)

# Encode 'Embarked' column
train_data = pd.get_dummies(train_data, columns=['Embarked'], drop_first=True)

# Encode 'Title' column
train_data = pd.get_dummies(train_data, columns=['Title'], drop_first=True)
```

### Step 6: Drop Unnecessary Columns
Remove columns that are not useful for the analysis.
** Dropping unnecessary columns helps reduce noise and focus on the most relevant features for analysis and modeling. **

```
train_data.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)
```

### Step 7: Normalize Numerical Features
Scale numerical features to ensure they have similar ranges.
** Normalizing numerical features ensures that they have similar scales, which can improve the performance of many machine learning algorithms. **

```python
from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Scale the 'Age' and 'Fare' columns
train_data[['Age', 'Fare']] = scaler.fit_transform(train_data[['Age', 'Fare']])
```

### Step 8: Split the Data
Split the dataset into features and target variable.
** Splitting the data into features and target variable is essential for training and evaluating machine learning models. **

```python
# Define features and target variable
X = train_data.drop(columns=['Survived'])
y = train_data['Survived']
```

### Step 9: Save the Cleaned Data
Save the cleaned and preprocessed data for future use.
** Saving the cleaned data allows you to reuse it without repeating the preprocessing steps, making your workflow more efficient. **

```python
# Save the cleaned data to a new CSV file
train_data.to_csv('cleaned_train_data.csv', index=False)
```

## Conclusion
This project provides a comprehensive guide to data cleaning and preprocessing using the Titanic dataset. By following these steps, you can ensure that your data is well-prepared for further analysis and modeling.
