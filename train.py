import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn

from model import DeNoiser

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--alpha', type=float, default=0.1, help="fraction of original image size to use as latent dim")
    parser.add_argument('--batch_size', type=int, default=20, help="batch size")
    parser.add_argument('--epochs', type=int, default=10, help="number of epochs to run on training")
    parser.add_argument('--noise_var', type=float, default=0.5, help="variance of gausian noise")
    parser.add_argument('--valid_split', type=float, default=0.2, help="part of dataset to use as validation set")
    parser.add_argument('--loss', type=str, default='mse', help="loss function to use for training: BCE or MSE")
    parser.add_argument('--plot_imgs', action='store_true', help="plots a random image in each epoch")
    parser.add_argument('--model_file', type=str, default='mnist_autoencoder.pt', help="trained model path")
    return parser.parse_args()


def split_train_valid(dataset, batch_size, valid_split, shuffle_dataset = True,random_seed= 42):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(valid_split * dataset_size))
    if shuffle_dataset :
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
    return int(np.floor(h * w * alpha))


def add_noise(images, cnfg):
    ## add random noise to the input images
    noisy_imgs = images + cnfg.noise_var * torch.randn(*images.shape)
    # Clip the images to be between 0 and 1
    noisy_imgs = np.clip(noisy_imgs, 0., 1.)
    return noisy_imgs


def run_epoch(model, optimizer, criterion, train_loader, cnfg, e):
    train_loss = 0.0
    pbar = tqdm(train_loader)
    for data in pbar:
        images, _ = data
        noisy_imgs = add_noise(images, cnfg)
        optimizer.zero_grad()
        outputs = model(noisy_imgs)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)

    train_loss = train_loss / len(train_loader)
    print('Epoch: {}'.format(e),
          '\tTraining Loss: {:.4f}'.format(train_loss))
    return train_loss


def validate(model, criterion, valid_loader, cnfg, e):
    valid_loss = 0.0
    with torch.no_grad():
        model.eval()
        pbar = tqdm(valid_loader)
        for data in pbar:
            images, _ = data
            noisy_imgs = add_noise(images, cnfg)
            outputs = model(noisy_imgs)
            loss = criterion(outputs, images)
            valid_loss += loss.item() * images.size(0)
    valid_loss = valid_loss / len(valid_loader)
    print('Epoch: {}'.format(e),
          '\tTraining Loss: {:.4f}'.format(valid_loss))
    return valid_loss


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

    train_losses, valid_losses = list(), list()
    for e in range(1, cnfg.epochs + 1):
        train_loss = run_epoch(model, optimizer, criterion, train_loader, cnfg, e)
        train_losses.append(train_loss)
        valid_loss = validate(model, criterion, valid_loader, cnfg, e)
        valid_losses.append(valid_loss)

    plot_epochs_loss(train_losses, valid_losses)

    return valid_loss


def main():
    args = parse_args()
    valid_loss = train(args)


if __name__ == '__main__':
    main()
