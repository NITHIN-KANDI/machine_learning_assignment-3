import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def read_data(file_path):
    """Read data from an Excel file and return a DataFrame."""
    return pd.read_excel(file_path)

def compute_individual_costs(features, target):
    """Compute individual costs using linear regression."""
    pinv_features = np.linalg.pinv(features)
    return pinv_features @ target

def classify_customers(data):
    """Classify customers into 'RICH' or 'POOR' categories based on total payment."""
    data['Category'] = data['Payment (Rs)'].apply(lambda x: 'RICH' if x > 200 else 'POOR')
    return data

def train_logistic_regression(data):
    """Train a logistic regression classifier."""
    features = data[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']]
    target = data['Category']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    data['Predicted Category'] = classifier.predict(features)
    return data

def main():
    # Read data
    dataset = read_data('Lab Session1 Data.xlsx')

    # Display the first few rows of the dataset
    print(dataset.head())

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

    # Compute individual costs using linear regression
    individual_costs = compute_individual_costs(A, C)
    print("The individual cost of a candy is: ", round(individual_costs[0][0]))
    print("The individual cost of a mango is: ", round(individual_costs[1][0]))
    print("The individual cost of a milk packet is: ", round(individual_costs[2][0]))

    # Load data again into a pandas DataFrame
    df = read_data('Lab Session1 Data.xlsx')

    # Classify customers into 'RICH' or 'POOR' categories
    df = classify_customers(df)

    # Apply logistic regression classifier to predict categories
    df = train_logistic_regression(df)

    # Display selected columns from the DataFrame
    print(df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)', 'Category', 'Predicted Category']])

if __name__ == "__main__":
    main()
