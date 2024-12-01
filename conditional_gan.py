# This files serves the neccessary functions for generating images using pretrained models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from models import get_noise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def display_image_grid(images, num_rows=5, title=""):
    if(images.shape[-1]!=28):
        images = images.view(-1, 1, 28, 28)
    plt.figure(figsize=(5, 5))
    plt.axis("off")
    plt.title(title)
    grid = make_grid(images.detach().cpu()[:25], nrow=num_rows).permute(1, 2, 0).numpy()
    plt.imshow(grid)
    plt.show()

def check_generation(generator):
    generator.eval()
    labels = torch.tensor([0,1,2,3,4,5,6,7,8,9] * 10).to(device)
    fake_eval_batch = generator(get_noise(100, 10, device=device), labels).view(-1, 1, 28, 28)
    grid = make_grid(fake_eval_batch.detach().cpu(), nrow=10).permute(1, 2, 0).numpy()
    plt.figure(figsize=(9, 9))
    plt.title("Generated Images")
    plt.axis('off')
    plt.xlabel("Class")
    plt.imshow(grid)
    plt.show()

def generate_digit(generator, digit):
    generator.eval()
    labels = torch.tensor([digit] * 25).to(device)
    fake_eval_batch = generator(get_noise(25, 10, device=device), labels).view(-1, 1, 28, 28)
    grid = make_grid(fake_eval_batch.detach().cpu(), nrow=5).permute(1, 2, 0).numpy()
    plt.figure(figsize=(5, 5))
    # no border
    plt.axis('off')
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(grid)
    plt.savefig('generated_digit.png', bbox_inches='tight', pad_inches=0)  # Save the generated image
    return 'generated_digit.png'  # Return the image path