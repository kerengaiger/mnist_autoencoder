import torch.nn as nn
import torch.nn.functional as F


class DeNoiser(nn.Module):
    def __init__(self, latent_dim):
        super(DeNoiser, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Finish encoder with linear layer depend on laten_dim
        self.fc1 = nn.Linear(8 * 3 * 3, latent_dim)

        # Decoder
        self.fc2 = nn.Linear(latent_dim, 8 * 3 * 3)

        self.t_conv1 = nn.ConvTranspose2d(8, 8, 3, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 32, 2, stride=2)
        # one, final, normal conv layer to decrease the depth
        self.conv_out = nn.Conv2d(32, 1, 3, padding=1)

    def encoder(self, x):
        # 28 * 28 -> 14 * 14
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # 14 * 14 -> 7 * 7
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # 7 * 7 -> 3 * 3
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        h, w = x.shape[2], x.shape[3]
        x = x.reshape(x.shape[0], -1)
        # batch_size * (8*3*3) -> batch_size * hidden†_
        x = self.fc1(x)
        return x, h, w

    def decoder(self, x, h, w):
        x = self.fc2(x)
        x = x.view(x.shape[0], -1, h, w)

        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        # transpose again, output should have a sigmoid applied
        x = F.sigmoid(self.conv_out(x))
        return x

    def forward(self, x):
        x, h, w = self.encoder(x)
        x = self.decoder(x, h, w)
        return x
