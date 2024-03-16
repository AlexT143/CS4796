import os
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np  # For the square transformation

# Create the "visualizations" folder if it doesn't exist
visualizations_folder = "visualizations"
if not os.path.exists(visualizations_folder):
    os.makedirs(visualizations_folder)

# Read the data from the CSV file
csv_file = os.path.join("models_data", "vanilla_transformer_hyperparameters_results.csv")
data = pd.read_csv(csv_file)

volatility_types = data["volatility_type"].unique()

# Add an identifier to the data
data['id'] = range(len(data))

# Hyperparameters to include in t-SNE
hyperparameters = ["d_model", "num_heads", "dff", "dropout_rate"]

# Prepare data for t-SNE
hyperparameter_data = data[hyperparameters].values

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, perplexity=100, learning_rate=200, random_state=0)
reduced_data = tsne.fit_transform(hyperparameter_data)

# Square the loss for transformation
data['loss_squared'] = data['loss']**(1/4)

# Combine the reduced data with the original data for easy plotting
combined_data = pd.concat([pd.DataFrame(reduced_data, columns=['dim1', 'dim2']), data[['loss', 'loss_squared', 'volatility_type', 'id']]], axis=1)


# Define a distance threshold for "near (50, 55)"
threshold_distance = 10  # Adjust based on your plot scale and the density of the cluster

for volatility_type in volatility_types:
    # Filter for the current volatility type
    filtered_data = combined_data[combined_data['volatility_type'] == volatility_type]

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(filtered_data['dim1'], filtered_data['dim2'], c=filtered_data['loss_squared'], cmap='viridis')

    # Colorbar and labels
    cbar = plt.colorbar(scatter)
    cbar.set_label('Squared Loss')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(f't-SNE of Transformer Hyperparameters - {volatility_type}')

    # If the current plot is for the 'involatile' volatility type, perform further analysis
    if volatility_type == 'inline':
        # Filter points within the threshold distance from (50, 55)
        cluster_points = filtered_data[
            np.sqrt((filtered_data['dim1'] - 40)**2 + (filtered_data['dim2'] - 55)**2) <= threshold_distance]
        
        # Use identifiers to find the corresponding hyperparameters
        cluster_configurations = data.loc[data['id'].isin(cluster_points['id'])]
        
        # Display or analyze the configurations
        print(f"Configurations near (50, 55) for 'involatile':\n",
              cluster_configurations[['id', 'd_model', 'num_heads', 'dff', 'dropout_rate', 'loss']])
    if volatility_type == 'volatile':
        # Filter points within the threshold distance from (50, 55)
        cluster_points = filtered_data[
            np.sqrt((filtered_data['dim1'] - 90)**2 + (filtered_data['dim2'] - 150)**2) <= threshold_distance]
        
        # Use identifiers to find the corresponding hyperparameters
        cluster_configurations = data.loc[data['id'].isin(cluster_points['id'])]
        
        # Display or analyze the configurations
        print(f"Configurations near (50, 55) for 'volatile':\n",
              cluster_configurations[['id', 'd_model', 'num_heads', 'dff', 'dropout_rate', 'loss']])
    if volatility_type == 'involatile':
        # Filter points within the threshold distance from (50, 55)
        cluster_points = filtered_data[
            np.sqrt((filtered_data['dim1'] + 50)**2 + (filtered_data['dim2'] - 100)**2) <= threshold_distance]
        
        # Use identifiers to find the corresponding hyperparameters
        cluster_configurations = data.loc[data['id'].isin(cluster_points['id'])]
        
        # Display or analyze the configurations
        print(f"Configurations near (50, 55) for 'near':\n",
              cluster_configurations[['id', 'd_model', 'num_heads', 'dff', 'dropout_rate', 'loss']])
    
    # Save the plot
    plot_file = os.path.join(visualizations_folder, f"tsne_hyperparameters_squared_loss_{volatility_type}.png")
    plt.savefig(plot_file)
    print(f"t-SNE plot with squared loss for {volatility_type} saved as: {plot_file}")

    plt.close(fig)  # Close the figure
