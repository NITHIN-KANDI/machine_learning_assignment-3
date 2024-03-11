import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Read data from an Excel file
dataset = pd.read_excel('Lab Session1 Data.xlsx')

# Display the first few rows of the dataset
a = dataset.head()
print(a)

# Extract features (candies, mangoes, milk packets) and target variable (payment)
A = dataset[['Candies (#)','Mangoes (Kg)','Milk Packets (#)']].values
C = dataset[['Payment (Rs)']].values

# Display extracted features and target variable
print(A)
print(C)

# Get the dimensions of the feature matrix
rows, cols = A.shape
print("The Dimensionality of the vector space:", cols)
print("Number of vectors are:", rows)

# Compute the rank of the feature matrix
rank = np.linalg.matrix_rank(A)
print("The rank of matrix A:", rank)

# Compute the pseudo-inverse of A and calculate individual costs using linear regression
pinv_A = np.linalg.pinv(A)
X = pinv_A @ C
print("The individual cost of a candy is: ", round(X[0][0]))
print("The individual cost of a mango is: ", round(X[1][0]))
print("The individual cost of a milk packet is: ", round(X[2][0]))

# Define a function for classification using logistic regression
def classifier(df):
    # Define features and target variable
    features = ["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]
    X = df[features]
    y = df['Category']
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Initialize and train the logistic regression classifier
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    # Predict categories for all data points
    df['Predicted Category'] = classifier.predict(X)
    return df

# Load data again into a pandas DataFrame
df = pd.read_excel('Lab Session1 Data.xlsx')

# Create a new column 'Category' based on total payment
df['Category'] = df['Payment (Rs)'].apply(lambda x: 'RICH' if x > 200 else 'POOR')

# Apply the classifier function to classify customers
df = classifier(df)

# Display selected columns from the DataFrame
print(df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)', 'Category', 'Predicted Category']])
