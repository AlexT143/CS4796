import yfinance as yf
import numpy as np
from preprocessing_utils import (
    embargo_data,
    purge_data,
    scale_data,
    create_sequences,
    train_test_split,
)
from VanillaTransformerComponents.hyperparameters import SEQ_LENGTH

def preprocess_data(volatility_type="involatile", use_log_returns=False):
    # Set the common start date for all equities
    start_date = "2000-01-01"

    # Select equity based on volatility type
    if volatility_type == "volatile":
        symbol = "AMZN"  # Amazon (volatile)
    elif volatility_type == "involatile":
        symbol = "JNJ"  # Johnson & Johnson (involatile)
    else:  # "market"
        symbol = "SPY"  # S&P 500 ETF (in line with market)

    # Download stock data
    data = yf.download(symbol, start=start_date)

    if use_log_returns:
        # Calculate log returns
        log_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
        data = log_returns.to_frame(name='LogReturns')

    # Apply purging and embargoing
    # data = purge_data(data, 2) # Purge data older than 2 years
    data = embargo_data(data, SEQ_LENGTH) # Embargo the most recent 5 days

    # Scale data
    scaled_data, scaler = scale_data(data)

    # Create sequences
    X, y = create_sequences(scaled_data, SEQ_LENGTH)

    # Split into training and testing sets
    X_train, y_train, X_test, y_test, train_size = train_test_split(X, y, 0.8)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print("Train size:", train_size)

    return X_train, y_train, X_test, y_test, scaler, scaled_data, train_size