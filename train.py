import argparse
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from tqdm import tqdm

from model import DeNoiser
from utils import add_noise, plot_imgs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--alpha', type=float, default=0.5, help="fraction of original image size to use as latent dim")
    parser.add_argument('--batch_size', type=int, default=20, help="batch size")
    parser.add_argument('--epochs', type=int, default=10, help="number of epochs to run on training")
    parser.add_argument('--noise_var', type=float, default=0.5, help="variance of gausian noise")
    parser.add_argument('--valid_split', type=float, default=0.2, help="part of dataset to use as validation set")
    parser.add_argument('--loss', type=str, default='mse', help="loss function to use for training: BCE or MSE")
    parser.add_argument('--plot_imgs', action='store_true', help="plots the first epoch images in each epoch")
    parser.add_argument('--plot_kernels', action='store_true', help="plots the conv1 kernels in each epoch")
    parser.add_argument('--save_dir', type=str, default='./figures/', help="directory to store figures in case "
                                                                           "plot_imgs is configured")
    parser.add_argument('--model_file', type=str, default='mnist_autoencoder.pt', help="trained model path")
    return parser.parse_args()


def split_train_valid(dataset, batch_size, valid_split, shuffle_dataset=True, random_seed= 42):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(valid_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    return train_loader, validation_loader


def induce_latent_dim(h, w, alpha):
    # TODO - remove print
    print('latent_dim', int(np.floor(h * w * alpha)))
    return int(np.floor(h * w * alpha))


def plot_batch(noisy_imgs, outputs, save_dir, fig_name):
    plot_imgs(noisy_imgs, save_dir, f'{fig_name}_noisy')
    plot_imgs(outputs, save_dir, f'{fig_name}_clean')


def plot_kernel_map(model, input, e, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    input = torch.unsqueeze(input, 0)
    output = model(input)

    kernels = model.conv1.weight.detach()
    fig, axarr = plt.subplots(4, 8)
    i = 0
    for row in range(4):
        for ax in range(8):
            axarr[row][ax].imshow(kernels[i].squeeze(), cmap='gray')
            axarr[row][ax].get_xaxis().set_visible(False)
            axarr[row][ax].get_yaxis().set_visible(False)
            i += 1
    fig.savefig(pathlib.Path(save_dir, f'kernal_conv1_epoch_{e}.png'))


def run_epoch(model, optimizer, criterion, train_loader, cnfg, e, plot_imgs):
    train_loss = 0.0
    pbar = tqdm(train_loader)
    for data in pbar:
        images, _ = data
        noisy_imgs = add_noise(images, cnfg.noise_var)
        optimizer.zero_grad()
        outputs = model(noisy_imgs)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)

    train_loss = train_loss / len(train_loader)
    if cnfg.plot_imgs:
        noisy_imgs_plot = add_noise(plot_imgs, cnfg.noise_var)
        outputs_plot = model(noisy_imgs_plot)
        plot_batch(noisy_imgs_plot, outputs_plot, cnfg.save_dir, f'epoch_{e}')

    if cnfg.plot_kernels:
        plot_kernel_map(model, plot_imgs[0], e, cnfg.save_dir)
    return train_loss


def validate(model, eval_loader, cnfg):
    criterion = nn.MSELoss()
    eval_loss = 0.0
    with torch.no_grad():
        model.eval()
        pbar = tqdm(eval_loader)
        for data in pbar:
            images, _ = data
            noisy_imgs = add_noise(images, cnfg.noise_var)
            outputs = model(noisy_imgs)
            loss = criterion(outputs, images)
            eval_loss += loss.item() * images.size(0)
    eval_loss = eval_loss / len(eval_loader)

    return eval_loss


def plot_epochs_loss(train_losses, valid_losses):
    fig, ax = plt.subplots(constrained_layout=True)

    ax.plot(range(len(train_losses)), train_losses, label="train_loss")
    ax.plot(range(len(valid_losses)), valid_losses, label="valid_loss")
    ax.set_xlabel('epochs')
    ax.set_ylabel(r'MSE loss')

    plt.title('Train / Valid Loss per epoch')
    plt.legend()
    fig.savefig(f'plot_epochs.png')


def train(cnfg):
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
    train_loader, valid_loader = split_train_valid(train_data, cnfg.batch_size, valid_split=cnfg.valid_split,
                                                   shuffle_dataset=True, random_seed=42)
    orig_h, orig_w = next(iter(train_loader))[0].shape[2], next(iter(train_loader))[0].shape[3]
    model = DeNoiser(induce_latent_dim(orig_h, orig_w, cnfg.alpha))

    if cnfg.loss == 'mse':
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), cnfg.lr)

    batch_imgs_plot = next(iter(train_loader))[0]

    train_losses, valid_losses = list(), list()
    for e in range(1, cnfg.epochs + 1):
        train_loss = run_epoch(model, optimizer, criterion, train_loader, cnfg, e, batch_imgs_plot)
        train_losses.append(train_loss)
        print('Epoch: {}'.format(e),
              '\tTraining Loss: {:.4f}'.format(train_loss))
        valid_loss = validate(model, valid_loader, cnfg)
        print('Epoch: {}'.format(e),
              '\tValidation Loss: {:.4f}'.format(valid_loss))
        valid_losses.append(valid_loss)

    plot_epochs_loss(train_losses, valid_losses)
    torch.save(model, cnfg.model_file)

    return model


def main():
    args = parse_args()
    model = train(args)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)
    test_loss = validate(model, test_loader, args)
    print(f'Test reconstruction Loss: {test_loss}')


if __name__ == '__main__':
    main()
