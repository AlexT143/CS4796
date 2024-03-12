import os
import sys
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from data_preprocessing import preprocess_data
from Baselines.TrainRNNBaseline import make_predictions, reshape_data

def test_rnn():
    # Preprocess the data
    X_train, y_train, X_test, y_test, scaler, scaled_data, train_size = preprocess_data()

    # Load the saved RNN model
    model_dir = "models/RNN"
    rnn_model = load_model(f"{model_dir}/rnn_model.h5")

    # Reshape the test data for RNN input
    _, X_test = reshape_data(X_train, X_test)

    # Make predictions using the loaded RNN model
    rnn_predictions = make_predictions(rnn_model, X_test)

    # Inverse scale the predictions and test data
    y_test_inverse = scaler.inverse_transform(y_test)
    rnn_predictions_inverse = scaler.inverse_transform(rnn_predictions)

    # Select the close price from the inverse scaled data
    y_test_close_inverse = y_test_inverse[:, 3]
    rnn_predictions_close_inverse = rnn_predictions_inverse[:, 3]

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test_close_inverse, rnn_predictions_close_inverse)
    mae = mean_absolute_error(y_test_close_inverse, rnn_predictions_close_inverse)
    mape = mean_absolute_percentage_error(y_test_close_inverse, rnn_predictions_close_inverse)

    print("Evaluation Metrics for RNN:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}")

if __name__ == "__main__":
    test_rnn()