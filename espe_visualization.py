import matplotlib.pyplot as plt
from VanillaTransformerComponents.utils import exponential_smoothing_positional_encoding

# Parameters
max_position = 50
d_model = 128
alpha = 0.1

# Generate exponential smoothing positional encoding
espe = exponential_smoothing_positional_encoding(max_position, d_model, alpha)

# Plot the exponential smoothing positional encoding
plt.figure(figsize=(10, 6))
plt.imshow(espe.numpy()[0], cmap='viridis', aspect='auto')
plt.xlabel('Depth')
plt.ylabel('Position')
plt.colorbar(label='Value')
plt.title(f'Exponential Smoothing Positional Encoding (alpha={alpha})')
plt.show()