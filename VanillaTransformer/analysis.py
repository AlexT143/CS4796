import os
import pandas as pd

# Read the data from the CSV file
csv_file = os.path.join("models_data", "vanilla_transformer_structure_results.csv")
data = pd.read_csv(csv_file)

# Calculate the median and average loss for each combination of encoder and decoder layers
results = data.groupby(["num_encoder_layers", "num_decoder_layers"]).agg(
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

# Print the results
print("Best Model (Median Loss):")
print(f"Number of Encoder Layers: {best_model_median['num_encoder_layers']}")
print(f"Number of Decoder Layers: {best_model_median['num_decoder_layers']}")
print(f"Median Loss: {best_model_median['median_loss']}")
print()

print("Runner-up Model (Median Loss):")
print(f"Number of Encoder Layers: {runner_up_model_median['num_encoder_layers']}")
print(f"Number of Decoder Layers: {runner_up_model_median['num_decoder_layers']}")
print(f"Median Loss: {runner_up_model_median['median_loss']}")
print()

print("Best Model (Average Loss):")
print(f"Number of Encoder Layers: {best_model_avg['num_encoder_layers']}")
print(f"Number of Decoder Layers: {best_model_avg['num_decoder_layers']}")
print(f"Average Loss: {best_model_avg['average_loss']}")
print()

print("Runner-up Model (Average Loss):")
print(f"Number of Encoder Layers: {runner_up_model_avg['num_encoder_layers']}")
print(f"Number of Decoder Layers: {runner_up_model_avg['num_decoder_layers']}")
print(f"Average Loss: {runner_up_model_avg['average_loss']}")
print()

print("Model Counts:")
print(results[["num_encoder_layers", "num_decoder_layers", "count"]])