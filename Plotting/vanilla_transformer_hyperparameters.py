import os
import pandas as pd
import matplotlib.pyplot as plt

# Create the "visualizations" folder if it doesn't exist
visualizations_folder = "visualizations"
if not os.path.exists(visualizations_folder):
    os.makedirs(visualizations_folder)

# Read the data from the CSV file
csv_file = os.path.join("models_data", "vanilla_transformer_hyperparameters_results.csv")
data = pd.read_csv(csv_file)

# Get the unique volatility types
volatility_types = data["volatility_type"].unique()

# Get the hyperparameters to plot (excluding encoder layers, decoder layers, and run)
hyperparameters = ["d_model", "num_heads", "dff", "dropout_rate"]

# Create a separate plot for each hyperparameter and volatility type
for volatility_type in volatility_types:
    for hyperparameter in hyperparameters:
        # Filter the data for the current volatility type
        data_filtered = data[data["volatility_type"] == volatility_type]
        
        # Get the unique values of the current hyperparameter
        unique_values = sorted(data_filtered[hyperparameter].unique())
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create a box and whiskers plot for each unique value of the current hyperparameter
        box_data = [data_filtered[data_filtered[hyperparameter] == value]["loss"] for value in unique_values]
        ax.boxplot(box_data, labels=unique_values)
        
        # Set labels and title
        ax.set_xlabel(hyperparameter)
        ax.set_ylabel("Loss")
        ax.set_title(f"Vanilla Transformer Hyperparameter Results - {hyperparameter} - {volatility_type}")
        
        plt.tight_layout()
        
        # Save the plot in the "visualizations" folder
        plot_file = os.path.join(visualizations_folder, f"vanilla_transformer_hyperparameter_boxplot_{hyperparameter}_{volatility_type}.png")
        plt.savefig(plot_file)
        
        print(f"Box and whiskers plot visualization for {hyperparameter} - {volatility_type} saved as: {plot_file}")