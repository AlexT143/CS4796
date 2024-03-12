from pmdarima import auto_arima
import pickle
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from data_preprocessing import preprocess_data

def train_and_save_auto_arima(data, train_size, model_type):
    # Select the fourth column (index 3) for training the auto_arima model
    close_data = data[:, 3]

    # Train the auto_arima model based on the model type
    if model_type == "volatile":
        auto_arima_model = auto_arima(close_data[:train_size], max_p=3, max_q=1)
    elif model_type == "involatile":
        auto_arima_model = auto_arima(close_data[:train_size], max_p=1, max_q=3)
    else:  # model_type == "inline"
        auto_arima_model = auto_arima(close_data[:train_size], max_p=2, max_q=2)

    # Print the selected ARIMA model
    print(f"Selected ARIMA Model for {model_type}:", auto_arima_model.order)

    # Save the auto_arima model
    model_dir = f"models/{model_type}_auto_ARIMA"
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

# Train and save the auto_arima models for different market volatility levels
volatile_auto_arima_model = train_and_save_auto_arima(scaled_data, train_size, "volatile")
involatile_auto_arima_model = train_and_save_auto_arima(scaled_data, train_size, "involatile")
inline_auto_arima_model = train_and_save_auto_arima(scaled_data, train_size, "inline")

# Make predictions using the trained auto_arima models
volatile_auto_arima_predictions = make_predictions(volatile_auto_arima_model, scaled_data, train_size)
involatile_auto_arima_predictions = make_predictions(involatile_auto_arima_model, scaled_data, train_size)
inline_auto_arima_predictions = make_predictions(inline_auto_arima_model, scaled_data, train_size)