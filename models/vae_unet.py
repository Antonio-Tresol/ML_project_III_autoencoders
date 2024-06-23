import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.unet import Down, InConv, Up, Decoder

class Variational_Encoder(nn.Module):
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.n_channels = input_channels
        self.inc = InConv(input_channels, 64)
        self.down1 = Down(64, 128, stride=4)
        self.down2 = Down(128, 256, stride=4)
        self.down3 = Down(256, 512, stride=4)
        self.down4 = Down(512, 512, stride=4)

        kernel_size = 1

        # latent space
        self.mean1 = nn.Conv2d(64, 64, kernel_size=kernel_size)
        self.log_var1 = nn.Conv2d(64, 64, kernel_size=kernel_size)

        self.mean2 = nn.Conv2d(128, 128, kernel_size=kernel_size)
        self.log_var2 = nn.Conv2d(128, 128, kernel_size=kernel_size)

        self.mean3 = nn.Conv2d(256, 256, kernel_size=kernel_size)
        self.log_var3 = nn.Conv2d(256, 256, kernel_size=kernel_size)

        self.mean4 = nn.Conv2d(512, 512, kernel_size=kernel_size)
        self.log_var4 = nn.Conv2d(512, 512, kernel_size=kernel_size)

        self.mean5 = nn.Conv2d(512, 512, kernel_size=kernel_size)
        self.log_var5 = nn.Conv2d(512, 512, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> list [torch.Tensor]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Create latent space
        means = [self.mean1(x1), self.mean2(x2), self.mean3(x3), self.mean4(x4), self.mean5(x5)]
        log_vars = [self.log_var1(x1), self.log_var2(x2), self.log_var3(x3), self.log_var4(x4), self.log_var5(x5)]
        
        outputs = [means, log_vars]

        return outputs

class Variational_Decoder(nn.Module):
    def __init__(self, output_channels: int) -> None:
        super().__init__()
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, output_channels, kernel_size=1),
            nn.LeakyReLU()
        )

    def forward(self, encoder_outputs: torch.Tensor) -> torch.Tensor:
        x = encoder_outputs[-1]
        x = self.up1(x, encoder_outputs[-2])
        x = self.up2(x, encoder_outputs[-3])
        x = self.up3(x, encoder_outputs[-4])
        x = self.up4(x, encoder_outputs[-5])
        x = self.decoder(x)

        return x

class Reparameterizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, mean: list[torch.Tensor], log_var: list[torch.Tensor]) -> list[torch.Tensor]:
        epsilon = [ torch.randn_like(var) for var in log_var ]

        z = [ mean[i] + epsilon[i] * log_var[i] for i in range(len(log_var)) ]

        return z

class Variational_Unet(nn.Module):
    def __init__(self, in_channels: int, device: torch.device) -> None:
        super().__init__()
        self.n_channels = in_channels

        self.encoder = Variational_Encoder(input_channels = in_channels).to(device)
        self.reparameterizer = Reparameterizer().to(device)
        self.decoder = Decoder(output_channels = in_channels).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        means, log_vars = self.encoder(x)

        reparameterized = self.reparameterizer(means, log_vars)

        decoded = self.decoder(reparameterized) 

        return decoded


def vae_loss_fn(y_hat, y):
        return nn.MSELoss().forward(y_hat, y) + nn.KLDivLoss(reduction="batchmean").forward(y_hat, y)

