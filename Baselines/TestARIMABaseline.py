import pickle
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from data_preprocessing import preprocess_data
from Baselines.TrainARIMABaseline import make_predictions
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

def test_auto_arima():
    # Preprocess the data
    X_train, y_train, X_test, y_test, scaler, scaled_data, train_size = preprocess_data()

    # Load the saved auto_arima model
    model_dir = "models/auto_ARIMA"
    with open(f"{model_dir}/auto_arima_model.pkl", "rb") as file:
        auto_arima_model = pickle.load(file)

    # Make predictions using the loaded auto_arima model
    auto_arima_predictions = make_predictions(auto_arima_model, scaled_data, train_size)

    # Inverse scale the predictions and test data
    close_scaler = MinMaxScaler()
    close_scaler.min_, close_scaler.scale_ = scaler.min_[3], scaler.scale_[3]
    auto_arima_predictions_inverse = close_scaler.inverse_transform(auto_arima_predictions.reshape(-1, 1))
    y_test_close = scaled_data[train_size:, 3][:len(auto_arima_predictions)]  # Select the close data from the test set
    y_test_close_inverse = close_scaler.inverse_transform(y_test_close.reshape(-1, 1))

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test_close_inverse, auto_arima_predictions_inverse)
    mae = mean_absolute_error(y_test_close_inverse, auto_arima_predictions_inverse)
    mape = mean_absolute_percentage_error(y_test_close_inverse, auto_arima_predictions_inverse)

    print("Evaluation Metrics for Auto ARIMA:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}")

if __name__ == "__main__":
    test_auto_arima()