import torch
import numpy as np
import matplotlib.pyplot as plt


def add_noise(images, noise_var):
    # add random noise to the input images
    noisy_imgs = images + noise_var * torch.randn(*images.shape)
    # Clip the images to be between 0 and 1
    noisy_imgs = np.clip(noisy_imgs, 0., 1.)
    return noisy_imgs


def plot_imgs(imgs, fig_name):
    batch_size = imgs.shape[0]
    imgs = imgs.detach().numpy()
    imgs = imgs.view(batch_size, 1, 28, 28)

    fig, axes = plt.subplots(nrows=1, ncols=batch_size, figsize=(25, 4))
    for img, ax in zip(imgs, axes):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(np.squeeze(img), cmap='gray')
    fig.savefig(fig_name)
