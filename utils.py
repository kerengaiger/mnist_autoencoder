import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib


def add_noise(images, noise_var):
    # add random noise to the input images
    noisy_imgs = images + noise_var * torch.randn(*images.shape)
    # Clip the images to be between 0 and 1
    noisy_imgs = np.clip(noisy_imgs, 0., 1.)
    return noisy_imgs


def plot_imgs(imgs, save_dir, fig_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    batch_size = imgs.shape[0]
    imgs = imgs.detach().numpy()

    if batch_size == 1:
        plt.imsave(pathlib.Path(save_dir, fig_name), np.squeeze(imgs), cmap='gray')

    else:
        fig, axes = plt.subplots(nrows=1, ncols=batch_size, figsize=(25, 4))
        for img, ax in zip(imgs, axes):
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.imshow(np.squeeze(img), cmap='gray')

        fig.savefig(pathlib.Path(save_dir, fig_name))
