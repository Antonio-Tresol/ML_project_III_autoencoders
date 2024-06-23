import torch
import torch.nn as nn

from models.unet import Down, InConv, Decoder

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

class Reparameterizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, mean: list[torch.Tensor], log_var: list[torch.Tensor]) -> list[torch.Tensor]:
        vars = [ torch.exp(0.5 * log_var[i]) for i in range(len(log_var)) ]
        epsilon = [ torch.randn_like(var) for var in vars ]

        z = [ mean[i] + epsilon[i] * vars[i] for i in range(len(vars)) ]
        z = z[-1]
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

        return decoded, means[-1], log_vars[-1]


def vae_loss_fn(y_hat, y, mean, log_var):
    print(y_hat.shape, y.shape, mean.shape, log_var.shape)
    return nn.MSELoss().forward(y_hat, y) + torch.mean(-0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp(), dim = 1), dim = 0)

class Variational_Encoder_Trial(nn.Module):
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
        self.mean5 = nn.Conv2d(512, 512, kernel_size=kernel_size)
        self.log_var5 = nn.Conv2d(512, 512, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> list [torch.Tensor]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Create latent space
        mean = self.mean5(x5)
        log_var = self.log_var5(x5)
        outputs = [x1, x2, x3, x4, x5]

        return outputs, mean, log_var

class Reparameterizer_Trial(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, mean: list[torch.Tensor], log_var: list[torch.Tensor]) -> list[torch.Tensor]:
        var = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(var)

        z = mean + epsilon * var
        return z

class Variational_Unet_Trial(nn.Module):
    def __init__(self, in_channels: int, device: torch.device) -> None:
        super().__init__()
        self.n_channels = in_channels

        self.encoder = Variational_Encoder_Trial(input_channels = in_channels).to(device)
        self.reparameterizer = Reparameterizer_Trial().to(device)
        self.decoder = Decoder(output_channels = in_channels).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, mean, log_var = self.encoder(x)

        reparameterized = self.reparameterizer(mean, log_var)
        encoded[-1] = reparameterized
        
        decoded = self.decoder(encoded) 

        return decoded, mean, log_var