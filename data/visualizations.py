import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from PIL import Image
import numpy as np
import h5py
from datasets import MediumImagenetHDF5Dataset

dataset = MediumImagenetHDF5Dataset(img_size=96, split="test")

data_loader = DataLoader(dataset, batch_size=10, shuffle=True)

# Function to unnormalize the image (convert it back to the original format)
def unnormalize(image, mean, std):
    for c in range(3):
        image[c] = image[c] * std[c] + mean[c]
    return image


def visualize_images():
    images, labels = next(iter(data_loader))
    
    # Unnormalize each image in the batch
    for i in range(images.size(0)):
        images[i] = unnormalize(images[i], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    grid = torchvision.utils.make_grid(images, nrow=5)
    grid = grid.permute(1, 2, 0).numpy()
    
    plt.imshow(grid)
    plt.axis('off')
    plt.savefig("output_image2.png")

# Call the function to visualize the images
visualize_images()
