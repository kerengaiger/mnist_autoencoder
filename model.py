import torch
import torch.nn as nn
import torch.nn.functional as F


class DeNoiser(nn.Module):
    def __init__(self, latent_dim):
        super(DeNoiser, self).__init__()
        # Encoder
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Finish encoder with linear layer depend on laten_dim
        self.fc1 = nn.Linear(128 * 3 * 3, latent_dim)

        # Decoder
        self.fc2 = nn.Linear(latent_dim, 128 * 3 * 3)

        self.t_conv1 = nn.ConvTranspose2d(128, 128, 3, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        # one, final, normal conv layer to decrease the depth
        self.conv_out = nn.Conv2d(32, 1, 3, padding=1)

    def encoder(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

    def decoder(self, x):
        x = self.fc2(x)
        x = x.view(x.shape[0], -1, 3, 3)

        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        # transpose again, output should have a sigmoid applied
        x = torch.sigmoid(self.conv_out(x))
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
