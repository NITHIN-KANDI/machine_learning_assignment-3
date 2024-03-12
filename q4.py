import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def load_data(file_path):
    """Load data from an Excel file and return a DataFrame."""
    return pd.read_excel(file_path)

def calculate_statistics(data):
    """Calculate mean and variance of prices."""
    mean_price = data['Price'].mean()
    variance_price = data['Price'].var()
    return mean_price, variance_price

def calculate_probability_loss(data):
    """Calculate the probability of making a loss over the stock."""
    loss_probability = (data['Chg%'] < 0).mean()
    return loss_probability

def calculate_probability_profit_on_wednesday(data):
    """Calculate the probability of making a profit on Wednesday."""
    wednesday_data = data[data['Day'] == 'Wednesday']
    wednesday_profit_probability = (wednesday_data['Chg%'] > 0).mean()
    return wednesday_profit_probability

def calculate_conditional_probability_profit(data):
    """Calculate the conditional probability of making a profit, given that today is Wednesday."""
    wednesday_profit_probability = calculate_probability_profit_on_wednesday(data)
    wednesday_count = (data['Day'] == 'Wednesday').sum()
    total_count = len(data)
    conditional_probability = wednesday_profit_probability * total_count / wednesday_count
    return conditional_probability

def visualize_data(data):
    """Visualize Chg% vs Day of the Week using scatter plot."""
    plt.scatter(data['Day'], data['Chg%'])
    plt.xlabel('Day of the Week')
    plt.ylabel('Chg%')
    plt.title('Chg% vs Day of the Week')
    plt.show()

def main():
    # Load data
    dataset = load_data('irctc.xlsx')
    print(dataset.head())

    # Calculate statistics
    mean_price, variance_price = calculate_statistics(dataset)
    print("Mean price:", mean_price)
    print("Variance of price:", variance_price)

    # Check if data is available for Wednesdays and April
    if 'Wednesday' in dataset['Day'].values:
        wednesday_profit_probability = calculate_probability_profit_on_wednesday(dataset)
        print("Probability of making a profit on Wednesday:", wednesday_profit_probability)
    else:
        print("No data available for Wednesdays.")

    if 'Apr' in dataset['Month'].values:
        april_profit_probability = calculate_probability_profit_on_wednesday(dataset[dataset['Month'] == 'Apr'])
        print("Probability of making a profit in April:", april_profit_probability)
    else:
        print("No data available for April.")

    # Calculate probabilities
    loss_probability = calculate_probability_loss(dataset)
    print("Probability of making a loss over the stock:", loss_probability)

    conditional_probability = calculate_conditional_probability_profit(dataset)
    print("Conditional probability of making profit, given that today is Wednesday:", conditional_probability)

    # Visualize data
    visualize_data(dataset)

if __name__ == "__main__":
    main()
