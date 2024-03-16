import os
import pandas as pd
import matplotlib.pyplot as plt

# Create the "visualizations" folder if it doesn't exist
visualizations_folder = "visualizations"
if not os.path.exists(visualizations_folder):
    os.makedirs(visualizations_folder)

# Read the data from the CSV file
csv_file = os.path.join("models_data", "arma_transformer_results.csv")
data = pd.read_csv(csv_file)

# Get the unique volatility types
volatility_types = data["volatility_type"].unique()

# Create a separate heat map for each volatility type
for volatility_type in volatility_types:
    # Filter the data for the current volatility type
    data_filtered = data[data["volatility_type"] == volatility_type]
    
    # Group by ar_order and ma_order and calculate the mean loss
    loss_grid = data_filtered.groupby(["ar_order", "ma_order"])["loss"].median().unstack()
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create a colormap
    cmap = plt.cm.viridis
    
    # Plot the heat map
    im = ax.imshow(loss_grid, cmap=cmap, aspect='auto', origin='lower')
    
    # Set tick labels for x and y axes
    ar_orders = loss_grid.columns.tolist()
    ma_orders = loss_grid.index.tolist()
    ax.set_xticks(range(len(ma_orders)))
    ax.set_xticklabels(ma_orders)
    ax.set_yticks(range(len(ar_orders)))
    ax.set_yticklabels(ar_orders)
    
    # Set labels and title
    ax.set_xlabel("MA Order")
    ax.set_ylabel("AR Order")
    ax.set_title(f"ARMA-Inspired Transformer Results (Median Loss) - {volatility_type}")
    
    # Add a colorbar to show the mapping of loss values to colors
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Median Loss")
    
    plt.tight_layout()
    
    # Save the plot in the "visualizations" folder
    plot_file = os.path.join(visualizations_folder, f"arma_transformer_heatmap_avg_{volatility_type}.png")
    plt.savefig(plot_file)
    
    print(f"Heat map visualization with median loss for {volatility_type} saved as: {plot_file}")
