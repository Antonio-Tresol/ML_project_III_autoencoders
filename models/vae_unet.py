import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.unet import Down, InConv, Up, Decoder

class Variational_Encoder(nn.Module):
    def __init__(self, input_channels: int, latent_dim: int = 256) -> None:
        super().__init__()
        self.n_channels = input_channels
        self.inc = InConv(input_channels, 64)
        self.down1 = Down(64, 128, stride=4)
        self.down2 = Down(128, 256, stride=4)
        self.down3 = Down(256, 512, stride=4)
        self.down4 = Down(512, 512, stride=4)

        # latent space
        self.mean1 = nn.Conv2d(64, latent_dim, kernel_size=1)
        self.log_var1 = nn.Conv2d(64, latent_dim, kernel_size=1)

        self.mean2 = nn.Conv2d(128, latent_dim, kernel_size=1)
        self.log_var2 = nn.Conv2d(128, latent_dim, kernel_size=1)

        self.mean3 = nn.Conv2d(256, latent_dim, kernel_size=1)
        self.log_var3 = nn.Conv2d(256, latent_dim, kernel_size=1)

        self.mean4 = nn.Conv2d(512, latent_dim, kernel_size=1)
        self.log_var4 = nn.Conv2d(512, latent_dim, kernel_size=1)

        self.mean5 = nn.Conv2d(512, latent_dim, kernel_size=1)
        self.log_var5 = nn.Conv2d(512, latent_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        

        # Create latent space
        r1 = self.mean1(x1), self.log_var1(x1)
        r2 = self.mean2(x2), self.log_var2(x2)
        r3 = self.mean3(x3), self.log_var3(x3)
        r4 = self.mean4(x4), self.log_var4(x4)
        r5 = self.mean5(x5), self.log_var5(x5)


        outputs = [r1, r2, r3, r4, r5]

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

    def forward(self, mu: torch.Tensor, mean, log_var) -> list[torch.Tensor]:
        '''
        z = []

        for _ in range(len(logvar)):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)

            z_i = mu + eps * std

            z.append(z_i)

        '''

        epsilon = torch.randn_like(log_var)

        z = mean + epsilon * log_var

        return z

class Variational_Unet(nn.Module):
    def __init__(self, in_channels: int, device: torch.device) -> None:
        super().__init__()
        self.n_channels = in_channels

        self.encoder = Variational_Encoder(input_channels = in_channels).to(device)
        self.reparameterizer = Reparameterizer().to(device)
        self.decoder = Decoder(output_channels = in_channels).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, log_var = self.encoder(x)

        reparameterized = self.reparameterizer(mean, log_var)

        decoded = self.decoder(reparameterized) 

        return decoded