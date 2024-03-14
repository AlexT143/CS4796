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

    # Group by num_encoder_layers and num_decoder_layers and calculate the mean loss
    loss_grid = data_filtered.groupby(["num_decoder_layers", "num_encoder_layers"])["loss"].mean().unstack()

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create a colormap
    cmap = plt.cm.viridis

    # Plot the heat map
    im = ax.imshow(loss_grid, cmap=cmap, aspect='auto')

    # Set tick labels for x and y axes
    num_encoder_layers = loss_grid.columns.tolist()
    num_decoder_layers = loss_grid.index.tolist()
    ax.set_xticks(range(len(num_encoder_layers)))
    ax.set_xticklabels(num_encoder_layers)
    ax.set_yticks(range(len(num_decoder_layers)))
    ax.set_yticklabels(num_decoder_layers)

    # Set labels and title
    ax.set_xlabel("Number of Encoder Layers")
    ax.set_ylabel("Number of Decoder Layers")
    ax.set_title(f"Vanilla Transformer Structure Results (Average Loss) - {volatility_type}")

    # Add a colorbar to show the mapping of loss values to colors
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Average Loss")

    plt.tight_layout()

    # Save the plot in the "visualizations" folder
    plot_file = os.path.join(visualizations_folder, f"vanilla_transformer_structure_heatmap_avg_{volatility_type}.png")
    plt.savefig(plot_file)

    print(f"Heat map visualization with average loss for {volatility_type} saved as: {plot_file}")