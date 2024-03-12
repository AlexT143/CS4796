import matplotlib.pyplot as plt
from VanillaTransformerComponents.utils import positional_encoding

# Parameters
max_position = 50
d_model = 128

# Generate positional encoding
pos_encoding = positional_encoding(max_position, d_model)

# Plot the positional encoding
plt.figure(figsize=(10, 6))
plt.imshow(pos_encoding.numpy()[0], cmap='viridis', aspect='auto')
plt.xlabel('Depth')
plt.ylabel('Position')
plt.colorbar(label='Value')
plt.title('Positional Encoding')
plt.show()