import os
import pandas as pd
import matplotlib.pyplot as plt

# Create the "visualizations" folder if it doesn't exist
visualizations_folder = "visualizations"
if not os.path.exists(visualizations_folder):
    os.makedirs(visualizations_folder)

# Read the data from the CSV file
csv_file = os.path.join("models_data", "arima_transformer_results.csv")
data = pd.read_csv(csv_file)

# Create a scatter plot
fig, ax = plt.subplots(figsize=(8, 6))

# Create a colormap based on the loss values
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=data['loss'].min(), vmax=data['loss'].max())

# Create a ScalarMappable instance to map loss values to colors
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Plot the scatter points with colors based on loss values
scatter = ax.scatter(data["ar_order"], data["ma_order"],
                     c=data['loss'], cmap=cmap, alpha=0.7)

ax.set_xlabel("AR Order")
ax.set_ylabel("MA Order")
ax.set_title("ARIMA-Inspired Transformer Results")

# Add a colorbar to show the mapping of loss values to colors
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("Loss")

plt.tight_layout()

# Save the plot in the "visualizations" folder
plot_file = os.path.join(visualizations_folder, "arima_transformer_plot.png")
plt.savefig(plot_file)

print(f"Scatter plot visualization saved as: {plot_file}")