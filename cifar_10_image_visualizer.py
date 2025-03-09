import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to load a CIFAR-10 batch


def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as file:
        batch = pickle.load(file, encoding='bytes')
    return batch


# Path to the batch file (update the path as needed)
batch_path = "cifar-10-batches-py/data_batch_1"

# Load the batch
batch = load_cifar10_batch(batch_path)

# Extract images
images = batch[b'data']

# Reshape the first image to (32, 32, 3)
image = images[0].reshape(3, 32, 32).transpose(1, 2, 0)

# Ensure the directory exists
os.makedirs("data", exist_ok=True)

# Save the image
plt.imsave("data/img.png", image)

print("Image saved as data/img.png")
