import os
import pandas as pd

# Read the data from the CSV file
csv_file = os.path.join("models_data", "vanilla_transformer_hyperparameters_results.csv")
data = pd.read_csv(csv_file)

# Group the data by volatility type
volatility_types = data["volatility_type"].unique()

for volatility_type in volatility_types:
    print(f"Analyzing results for volatility type: {volatility_type}")
    
    # Filter the data for the current volatility type
    volatility_data = data[data["volatility_type"] == volatility_type]
    
    # Calculate the median and average loss for each combination of hyperparameters
    results = volatility_data.groupby(["d_model", "num_heads", "dff"]).agg(
        median_loss=("loss", "median"),
        average_loss=("loss", "mean"),
        count=("loss", "count")
    ).reset_index()
    
    # Find the best and runner-up models based on the median loss
    best_models_median = results.nsmallest(2, "median_loss")
    best_model_median = best_models_median.iloc[0]
    runner_up_model_median = best_models_median.iloc[1]
    
    # Find the best and runner-up models based on the average loss
    best_models_avg = results.nsmallest(2, "average_loss")
    best_model_avg = best_models_avg.iloc[0]
    runner_up_model_avg = best_models_avg.iloc[1]
    
    # Print the results for the current volatility type
    print("Best Model (Median Loss):")
    print(f"d_model: {best_model_median['d_model']}")
    print(f"num_heads: {best_model_median['num_heads']}")
    print(f"dff: {best_model_median['dff']}")
    print(f"Median Loss: {best_model_median['median_loss']}")
    print(f"Models Trained: {best_model_median['count']}")
    print()
    
    print("Runner-up Model (Median Loss):")
    print(f"d_model: {runner_up_model_median['d_model']}")
    print(f"num_heads: {runner_up_model_median['num_heads']}")
    print(f"dff: {runner_up_model_median['dff']}")
    print(f"Median Loss: {runner_up_model_median['median_loss']}")
    print(f"Models Trained: {runner_up_model_median['count']}")
    print()
    
    print("Best Model (Average Loss):")
    print(f"d_model: {best_model_avg['d_model']}")
    print(f"num_heads: {best_model_avg['num_heads']}")
    print(f"dff: {best_model_avg['dff']}")
    print(f"Average Loss: {best_model_avg['average_loss']}")
    print(f"Models Trained: {best_model_avg['count']}")
    print()
    
    print("Runner-up Model (Average Loss):")
    print(f"d_model: {runner_up_model_avg['d_model']}")
    print(f"num_heads: {runner_up_model_avg['num_heads']}")
    print(f"dff: {runner_up_model_avg['dff']}")
    print(f"Average Loss: {runner_up_model_avg['average_loss']}")
    print(f"Models Trained: {runner_up_model_avg['count']}")
    print()
    
    print("Model Counts:")
    print(results[["d_model", "num_heads", "dff", "count"]])
    print("\n" + "-" * 50 + "\n")