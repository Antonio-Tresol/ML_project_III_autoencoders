import torch
import torch.nn as nn
import configuration as config


class DoubleDownConv(nn.Module):
    """
    A module for performing a double convolution on a tensor.

    This module applies two sequential convolutions each followed by batch normalization and ReLU activation.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    output_channels : int
        Number of output channels.

    Attributes
    ----------
    conv : nn.Sequential
        Sequential container of two convolutional blocks.
    """

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 4, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DoubleConv module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor after applying two convolutions.
        """
        x = self.conv(x)
        return x


class DoubleUpConv(nn.Module):
    """
    A module for performing a double convolution on a tensor.

    This module applies two sequential convolutions each followed by batch normalization and ReLU activation.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    output_channels : int
        Number of output channels.
    out_padding : int
        Amount of padding to add to the output.

    Attributes
    ----------
    conv : nn.Sequential
        Sequential container of two convolutional blocks.
    """

    def __init__(
        self, input_channels: int, output_channels: int, out_padding: int = 0
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                input_channels, output_channels, 4, stride=2, output_padding=out_padding
            ),
            nn.ConvTranspose2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DoubleConv module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor after applying two convolutions.
        """
        x = self.conv(x)
        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 256, 12, 12)


class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=512, z_dim=32, device="cuda"):
        self.device = device
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            # image channels, 224, 224
            DoubleDownConv(input_channels=image_channels, output_channels=32),
            # 32, 111, 111
            DoubleDownConv(input_channels=32, output_channels=64),
            # 64, 54, 54
            DoubleDownConv(input_channels=64, output_channels=128),
            # 128, 26, 26
            DoubleDownConv(input_channels=128, output_channels=256),
            # 256, 12, 12
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=12,
                padding=0,
                stride=1,
                dilation=1,
            ),
            nn.Flatten(),
        )

        self.fully_connected_1 = nn.Linear(h_dim, z_dim)
        self.fully_connected_2 = nn.Linear(h_dim, z_dim)
        self.fully_connected_3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (512, 1, 1)),
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=12,
                padding=0,
                stride=1,
                dilation=1,
            ),
            # Result: 128, 26, 26
            DoubleUpConv(input_channels=256, output_channels=128),
            # Result: 64, 54, 54
            DoubleUpConv(input_channels=128, output_channels=64),
            # Result: 32, 110, 110
            DoubleUpConv(input_channels=64, output_channels=32, out_padding=1),
            # Result: image_channels, 224, 224
            DoubleUpConv(input_channels=32, output_channels=image_channels),
            # nn.Sigmoid(),
        )
        self.to(self.device)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fully_connected_2(h), self.fully_connected_2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        enconded_data = self.encoder(x)
        z, mu, logvar = self.bottleneck(enconded_data)
        z = self.fully_connected_3(z)
        decoded_data = self.decoder(z)
        return decoded_data, mu, logvar


def vae_loss_fn(y_hat, y, mean, log_var):
    """
    Calculate the loss for a Variational Autoencoder (VAE).

    This function computes the loss for a VAE, which is the sum of a reconstruction loss and the Kullback-Leibler divergence
    between the learned latent variable distribution and the prior distribution. The reconstruction loss measures how well
    the VAE can reconstruct the input data, while the Kullback-Leibler divergence acts as a regularizer.

    Parameters
    ----------
    y_hat : torch.Tensor
        The reconstructed input data.
    y : torch.Tensor
        The original input data.
    mean : torch.Tensor
        The mean of the latent variable distribution learned by the VAE.
    log_var : torch.Tensor
        The logarithm of the variance of the latent variable distribution learned by the VAE.

    Returns
    -------
    torch.Tensor
        The total loss for the VAE, which is the sum of the reconstruction loss (MSE) and the Kullback-Leibler divergence.
    """
    mse_loss = nn.MSELoss(reduction="sum")(y_hat, y) / config.BATCH_SIZE

    kl_loss = (
        -0.5
        * torch.sum(1 + log_var - mean.pow(2) - torch.exp(log_var))
        / config.BATCH_SIZE
    )

    return mse_loss + kl_loss
