import torch
import numpy as np


def add_noise(images, noise_var):
    # add random noise to the input images
    noisy_imgs = images + noise_var * torch.randn(*images.shape)
    # Clip the images to be between 0 and 1
    noisy_imgs = np.clip(noisy_imgs, 0., 1.)
    return noisy_imgs
