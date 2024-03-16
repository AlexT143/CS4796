import os
import pandas as pd
import matplotlib.pyplot as plt

# Create the "visualizations" folder if it doesn't exist
visualizations_folder = "visualizations"
if not os.path.exists(visualizations_folder):
    os.makedirs(visualizations_folder)

# Read the data from the CSV file
csv_file = os.path.join("models_data", "vanilla_transformer_structure_results.csv")
data = pd.read_csv(csv_file)

# Get the unique volatility types
volatility_types = data["volatility_type"].unique()

# Create a separate heat map for each volatility type
for volatility_type in volatility_types:
    # Filter the data for the current volatility type
    data_filtered = data[data["volatility_type"] == volatility_type]
    
    # Filter the data for encoder layers in the range 0-4
    data_filtered = data_filtered[data_filtered["num_encoder_layers"] <= 4]
    data_filtered = data_filtered[data_filtered["num_decoder_layers"] <= 4]
    
    # Group by num_encoder_layers and num_decoder_layers and calculate the mean training time
    time_grid = data_filtered.groupby(["num_encoder_layers", "num_decoder_layers"])["training_time"].mean().unstack()
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create a colormap
    cmap = plt.cm.viridis
    
    # Plot the heat map
    im = ax.imshow(time_grid, cmap=cmap, aspect='auto', origin='lower')
    
    # Set tick labels for x and y axes
    num_encoder_layers = time_grid.columns.tolist()
    num_decoder_layers = time_grid.index.tolist()
    ax.set_xticks(range(len(num_decoder_layers)))
    ax.set_xticklabels(num_decoder_layers)
    ax.set_yticks(range(len(num_encoder_layers)))
    ax.set_yticklabels(num_encoder_layers)
    
    # Set labels and title
    ax.set_xlabel("Number of Decoder Layers")
    ax.set_ylabel("Number of Encoder Layers")
    ax.set_title(f"Vanilla Transformer Structure Results (Average Training Time) - {volatility_type}")
    
    # Add a colorbar to show the mapping of training time values to colors
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Average Training Time (seconds)")
    
    plt.tight_layout()
    
    # Save the plot in the "visualizations" folder
    plot_file = os.path.join(visualizations_folder, f"vanilla_transformer_structure_heatmap_time_{volatility_type}.png")
    plt.savefig(plot_file)
    
    print(f"Heat map visualization with average training time for {volatility_type} saved as: {plot_file}")