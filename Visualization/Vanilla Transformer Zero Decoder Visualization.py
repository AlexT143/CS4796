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

# Create a separate plot for each volatility type
for volatility_type in volatility_types:
    # Filter the data for options with 0 encoders and the current volatility type
    data_filtered = data[(data["num_encoder_layers"] == 0) & (data["volatility_type"] == volatility_type)]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create a box and whiskers plot for each number of decoders
    ax.boxplot([data_filtered[data_filtered["num_decoder_layers"] == num_decoders]["loss"] for num_decoders in sorted(data_filtered["num_decoder_layers"].unique())])

    # Set labels and title
    ax.set_xlabel("Number of Decoders")
    ax.set_ylabel("Loss")
    ax.set_title(f"Vanilla Transformer Structure Results (0 Encoders) - {volatility_type}")

    # Set x-axis tick positions and labels
    ax.set_xticks(range(1, len(sorted(data_filtered["num_decoder_layers"].unique())) + 1))
    ax.set_xticklabels(sorted(data_filtered["num_decoder_layers"].unique()))

    plt.tight_layout()

    # Save the plot in the "visualizations" folder
    plot_file = os.path.join(visualizations_folder, f"vanilla_transformer_structure_boxplot_0_encoders_{volatility_type}.png")
    plt.savefig(plot_file)
    print(f"Box and whiskers plot visualization with 0 encoders for {volatility_type} saved as: {plot_file}")