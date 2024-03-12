from pmdarima import auto_arima
import pickle
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from data_preprocessing import preprocess_data

def train_and_save_auto_arima(data, train_size):
    # Select the fourth column (index 3) for training the auto_arima model
    close_data = data[:, 3]

    # Train the auto_arima model
    auto_arima_model = auto_arima(close_data[:train_size])

    # Print the selected ARIMA model
    print("Selected ARIMA Model:", auto_arima_model.order)

    # Save the auto_arima model
    model_dir = "models/auto_ARIMA"
    os.makedirs(model_dir, exist_ok=True)
    with open(f"{model_dir}/auto_arima_model.pkl", "wb") as file:
        pickle.dump(auto_arima_model, file)

    return auto_arima_model

def make_predictions(auto_arima_model, data, train_size):
    # Select the fourth column (index 3) for making predictions
    close_data = data[:, 3]

    # Make predictions on the test set
    auto_arima_predictions = auto_arima_model.predict(n_periods=len(close_data) - train_size)

    return auto_arima_predictions

# Preprocess the data
X_train, y_train, X_test, y_test, scaler, scaled_data, train_size = preprocess_data()

# Train and save the auto_arima model
auto_arima_model = train_and_save_auto_arima(scaled_data, train_size)

# Make predictions using the trained auto_arima model
auto_arima_predictions = make_predictions(auto_arima_model, scaled_data, train_size)