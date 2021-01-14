import torch
import numpy as np


# Create a numpy array.
np_data = np.array([[1, 2], [3, 4]])

# Convert a numpy array to a torch tensor.
torch_data = torch.from_numpy(np_data)

# Convert a torch tensor to a numpy array.
np_data = torch_data.numpy()

# Convert a GPU torch tensor to a numpy array.
np_data = torch_data.data.cpu().numpy()

