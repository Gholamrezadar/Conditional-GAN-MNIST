import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn((n_samples, z_dim), device=device)

def get_random_labels(n_samples, device='cpu'):
    return torch.randint(0, 10, (n_samples,), device=device).type(torch.long)

def get_generator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True)
    )

class Generator(nn.Module):
    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super(Generator, self).__init__()

        # input is of shape (batch_size, z_dim + 10)
        self.gen = nn.Sequential(
            get_generator_block(z_dim + 10, hidden_dim), # 128
            get_generator_block(hidden_dim, hidden_dim*2), # 256
            get_generator_block(hidden_dim*2, hidden_dim*4), # 512
            get_generator_block(hidden_dim*4, hidden_dim*8), # 1024
            nn.Linear(hidden_dim*8, im_dim), # 784
            nn.Sigmoid(), # output between 0 and 1
        )

    def forward(self, noise, classes):
        '''
        noise (batch_size, z_dim) noise vector for each image in a batch 
        classes:long (batch_size) condition class for each image in a batch
        '''
        # classes = classes.type(torch.long)
        # one-hot encode condition_class e.g. 3 -> [0,0,0,1,0,0,0,0,0,0]
        one_hot_vec = F.one_hot(classes, num_classes=10).type(torch.float32) # (batch_size, 10)
        conditioned_noise = torch.concat((noise, one_hot_vec), dim=1) # (batch_size, z_dim + 10)
        return self.gen(conditioned_noise)


def get_discriminator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.2, inplace=True)
    )

class Discriminator(nn.Module):
    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim + 10, hidden_dim*4), # 512
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2), # 256
            get_discriminator_block(hidden_dim * 2, hidden_dim), # 128
            nn.Linear(hidden_dim, 1),
            # nn.Sigmoid(),
            # using a sigmoid followed by BCE is less numerically stable than BCEWithLogitsLoss alone
            # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss:~:text=This%20loss%20combines%20a%20Sigmoid%20layer%20and%20the%20BCELoss%20in%20one%20single%20class.%20This%20version%20is%20more%20numerically%20stable%20than%20using%20a%20plain%20Sigmoid%20followed%20by%20a%20BCELoss%20as%2C%20by%20combining%20the%20operations%20into%20one%20layer%2C%20we%20take%20advantage%20of%20the%20log%2Dsum%2Dexp%20trick%20for%20numerical%20stability.
        )

    def forward(self, image_batch):
        '''image_batch (batch_size, 784+10)'''
        return self.disc(image_batch)