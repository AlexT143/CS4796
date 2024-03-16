import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
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

# Hyperparameters to include in PCA
hyperparameters = ["d_model", "num_heads", "dff", "dropout_rate"]

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)  # Reduce to two dimensions for a 2D scatter plot
hyperparameter_data = data[hyperparameters]
reduced_data = pca.fit_transform(hyperparameter_data)

# Combine the reduced data with the original data for easy plotting
combined_data = np.hstack((reduced_data, data[['loss', 'volatility_type']].values))

# Create a separate plot for each volatility type
for volatility_type in volatility_types:
    # Filter the combined data for the current volatility type
    filtered_data = combined_data[data['volatility_type'] == volatility_type]

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(filtered_data[:, 0], filtered_data[:, 1], c=filtered_data[:, 2], cmap='viridis')

    # Colorbar and labels
    cbar = plt.colorbar(scatter)
    cbar.set_label('Loss')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'PCA of Transformer Hyperparameters - {volatility_type}')

    # Save the plot in the "visualizations" folder
    plot_file = os.path.join(visualizations_folder, f"pca_hyperparameters_{volatility_type}.png")
    plt.savefig(plot_file)
    print(f"PCA plot for {volatility_type} saved as: {plot_file}")

    plt.close(fig)  # Close the figure to avoid displaying it inline if running in a notebook
