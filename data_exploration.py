import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def check_stationarity(data):
    result = adfuller(data)
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    if result[1] <= 0.05:
        print("The data is stationary")
    else:
        print("The data is non-stationary")

def difference_data(data):
    diff_data = data.diff().dropna()
    return diff_data

def plot_data(data, title):
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()

def explore_data():
    # Download stock data
    data = yf.download("MCD", start="2010-01-01")

    # Plot the original data
    plot_data(data["Close"], "Original Data")

    # Check stationarity of the original data
    print("Stationarity of Original Data:")
    check_stationarity(data["Close"])

    # Apply differencing if the data is non-stationary
    diff_data = difference_data(data["Close"])

    # Plot the differenced data
    plot_data(diff_data, "Differenced Data")

    # Check stationarity of the differenced data
    print("Stationarity of Differenced Data:")
    check_stationarity(diff_data)

if __name__ == "__main__":
    explore_data()