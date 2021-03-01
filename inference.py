import torch
import argparse
import random

from torchvision import datasets
import torchvision.transforms as transforms

from utils import add_noise, plot_imgs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default='mnist_autoencoder.pt', help="trained model path")
    parser.add_argument('--save_dir', type=str, default='./figures/', help="directory to store figures")
    parser.add_argument('--noise_var', type=float, default=0.5, help="variance of gausian noise used in training")
    parser.add_argument('--imgs_to_gen', type=int, default=5, help="number of new images to generate")
    parser.add_argument('--imgs_to_denoise', type=int, default=5, help="number of sampled images to denoise")
    return parser.parse_args()


def img_denoising(img, model, noise_var, save_dir, i):
    img = torch.unsqueeze(img, 0)
    noisy_img = add_noise(img, noise_var)
    plot_imgs(noisy_img, save_dir, f'noisy_{i}.png')
    clean_img = model(noisy_img)
    plot_imgs(clean_img, save_dir, f'clean_{i}.png')


def generate_imgs(model, num_imgs, noise_var, save_dir):
    for i in range(num_imgs):
        img = torch.zeros(1, model.latent_dim)
        gaus_noise = add_noise(img, noise_var)
        new_img = model.decoder(gaus_noise)
        new_img = new_img.view(1, 1, 28, 28)
        plot_imgs(new_img, save_dir, f'new_image_{i}.png')


def main():
    args = parse_args()
    model = torch.load(args.model_file)
    # generate new images out of a random vector in the size of the latent dim
    generate_imgs(model, args.imgs_to_gen, args.noise_var, args.save_dir)
    # sample few images from MNIST dataset, add a Gaussian noise, then denoise it
    mnist_train = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
    sample_ids = random.sample(range(len(mnist_train)), args.imgs_to_denoise)
    sample_imgs = [mnist_train[i][0] for i in sample_ids]
    for i in range(len(sample_imgs)):
        img_denoising(sample_imgs[i], model, args.noise_var, args.save_dir, i)


if __name__ == '__main__':
    main()
