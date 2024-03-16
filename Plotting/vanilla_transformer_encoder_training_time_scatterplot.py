import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create the "visualizations" folder if it doesn't exist
visualizations_folder = "visualizations"
if not os.path.exists(visualizations_folder):
    os.makedirs(visualizations_folder)

# Read the data from the CSV file
csv_file = os.path.join("models_data", "vanilla_transformer_structure_results.csv")
data = pd.read_csv(csv_file)

# Filter the data for encoder layers in the range 0-4
data_filtered = data[data["num_encoder_layers"] <= 4]
data_filtered = data_filtered[data_filtered["num_decoder_layers"] <= 4]

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Create a scatter plot
ax.scatter(data_filtered["num_encoder_layers"], data_filtered["training_time"])

# Add regression line
z = np.polyfit(data_filtered["num_encoder_layers"], data_filtered["training_time"], 1)
p = np.poly1d(z)
plt.plot(data_filtered["num_encoder_layers"], p(data_filtered["num_encoder_layers"]), "black")

# Set labels and title
ax.set_xlabel("Number of Encoder Layers")
ax.set_ylabel("Training Time (seconds)")
ax.set_title("Vanilla Transformer Structure Results (Training Time)")

plt.tight_layout()

# Save the plot in the "visualizations" folder
plot_file = os.path.join(visualizations_folder, "vanilla_transformer_structure_scatter_plot_time.png")
plt.savefig(plot_file)

print(f"Scatter plot visualization with training time saved as: {plot_file}")