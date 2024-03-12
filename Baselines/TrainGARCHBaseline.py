import numpy as np
from arch import arch_model
import pickle
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from data_preprocessing import preprocess_data

def train_and_save_garch(data, train_size, p=1, q=1):
    # Create the GARCH model
    garch_model = arch_model(data[:train_size], p=p, q=q)
    
    # Fit the GARCH model
    garch_results = garch_model.fit()
    
    # Print the GARCH model summary
    print(garch_results.summary())
    
    # Save the GARCH model
    model_dir = "models/GARCH"
    os.makedirs(model_dir, exist_ok=True)
    with open(f"{model_dir}/garch_model.pkl", "wb") as file:
        pickle.dump(garch_results, file)
    
    return garch_results

def make_predictions(garch_results, data, train_size):
    # Get the estimated model parameters
    mu = garch_results.params['mu']
    omega = garch_results.params['omega']
    alpha = garch_results.params['alpha[1]']
    beta = garch_results.params['beta[1]']
    
    # Calculate the number of steps to forecast
    forecast_steps = len(data) - train_size
    
    # Initialize the forecast array
    garch_forecast = np.zeros(forecast_steps)
    
    # Set the initial values for the forecast
    garch_forecast[0] = mu
    sigma2_t = garch_results.conditional_volatility[-1] ** 2
    
    # Iterate over the forecast steps
    for t in range(1, forecast_steps):
        # Update the conditional variance (sigma2_t)
        sigma2_t = omega + alpha * garch_results.resid[-1] ** 2 + beta * sigma2_t
        
        # Calculate the forecast for the current step
        garch_forecast[t] = mu + np.sqrt(sigma2_t) * np.random.normal()
    
    return garch_forecast

# Preprocess the data with log returns
X_train, y_train, X_test, y_test, scaler, scaled_data, train_size = preprocess_data(use_log_returns=True)

# Train and save the GARCH model
garch_results = train_and_save_garch(scaled_data, train_size, p=1, q=1)

print("garch_results:", garch_results)
print(train_size)
print(scaled_data.shape)

# Make predictions using the trained GARCH model
garch_predictions = make_predictions(garch_results, scaled_data, train_size)