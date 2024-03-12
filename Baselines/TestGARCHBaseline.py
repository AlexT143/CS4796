import pickle
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from data_preprocessing import preprocess_data
from Baselines.TrainGARCHBaseline import make_predictions
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

def test_garch():
    # Preprocess the data with log returns
    X_train, y_train, X_test, y_test, scaler, scaled_data, train_size = preprocess_data(use_log_returns=True)
    
    # Load the saved GARCH model
    model_dir = "models/GARCH"
    with open(f"{model_dir}/garch_model.pkl", "rb") as file:
        garch_results = pickle.load(file)
    
    # Make predictions using the loaded GARCH model
    garch_predictions = make_predictions(garch_results, scaled_data, train_size)
   
    y_test_close = scaled_data[:len(garch_predictions)] 
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test_close, garch_predictions)
    mae = mean_absolute_error(y_test_close, garch_predictions)
    mape = mean_absolute_percentage_error(y_test_close, garch_predictions)
    
    print("Evaluation Metrics for GARCH:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}")

if __name__ == "__main__":
    test_garch()
    
    
#     import pickle
# import os
# import sys

# current_dir = os.path.dirname(os.path.abspath(__file__))
# root_dir = os.path.dirname(current_dir)
# sys.path.insert(0, root_dir)

# from data_preprocessing import preprocess_data
# from Baselines.TrainGARCHBaseline import make_predictions
# from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
# from sklearn.preprocessing import MinMaxScaler

# def test_garch():
#     # Preprocess the data
#     X_train, y_train, X_test, y_test, scaler, scaled_data, train_size = preprocess_data()
    
#     # Load the saved GARCH model
#     model_dir = "models/GARCH"
#     with open(f"{model_dir}/garch_model.pkl", "rb") as file:
#         garch_results = pickle.load(file)
    
#     # Make predictions using the loaded GARCH model
#     garch_predictions = make_predictions(garch_results, scaled_data, train_size)
    
#     # Inverse scale the predictions and test data
#     close_scaler = MinMaxScaler()
#     close_scaler.min_, close_scaler.scale_ = scaler.min_[3], scaler.scale_[3]
#     garch_predictions_inverse = close_scaler.inverse_transform(garch_predictions.reshape(-1, 1))
    
#     y_test_close = scaled_data[train_size:, 3][:len(garch_predictions)]  # Select the close data from the test set
#     y_test_close_inverse = close_scaler.inverse_transform(y_test_close.reshape(-1, 1))
    
#     # Calculate evaluation metrics
#     mse = mean_squared_error(y_test_close_inverse, garch_predictions_inverse)
#     mae = mean_absolute_error(y_test_close_inverse, garch_predictions_inverse)
#     mape = mean_absolute_percentage_error(y_test_close_inverse, garch_predictions_inverse)
    
#     print("Evaluation Metrics for GARCH:")
#     print(f"Mean Squared Error (MSE): {mse}")
#     print(f"Mean Absolute Error (MAE): {mae}")
#     print(f"Mean Absolute Percentage Error (MAPE): {mape}")

# if __name__ == "__main__":
#     test_garch()