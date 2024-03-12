import os
import sys
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

root_dir = os.path.dirname(parent_dir)
sys.path.insert(0, root_dir)

from data_preprocessing import preprocess_data

def test_base_transformer():
    # Preprocess the data
    X_train, y_train, X_test, y_test, scaler, scaled_data, train_size = preprocess_data()

    # Load the trained model
    model_dir = "models/decoder"
    transformer = tf.keras.models.load_model(f"{model_dir}/model")

    # Load the MinMaxScaler
    with open(f"{model_dir}/scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    # Make predictions using the loaded Transformer model
    transformer_predictions = transformer.predict(X_test)

    # Inverse scale the predictions and test data
    y_test_inverse = scaler.inverse_transform(y_test)
    transformer_predictions_inverse = scaler.inverse_transform(transformer_predictions)

    # Select the close price from the inverse scaled data
    y_test_close_inverse = y_test_inverse[:, 3]
    transformer_predictions_close_inverse = transformer_predictions_inverse[:, 3]

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test_close_inverse, transformer_predictions_close_inverse)
    mae = mean_absolute_error(y_test_close_inverse, transformer_predictions_close_inverse)
    mape = mean_absolute_percentage_error(y_test_close_inverse, transformer_predictions_close_inverse)

    print("Evaluation Metrics for Decoder:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}")

if __name__ == "__main__":
    test_base_transformer()