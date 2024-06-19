import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import configuration as config


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, stride: int):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2, stride=stride), DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.n_channels = in_channels
        self.inc = InConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        outputs = [x1, x2, x3, x4, x5]
        return outputs


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.down1 = Down(in_channels, 512)
        self.down2 = Down(512, 256)
        self.down3 = Down(256, 128)
        self.down4 = Down(128, 64)
        self.down5 = Down(64, 64)
        self.outc = OutConv(64, 1)

    def forward(self, encoder_outputs):
        encoder_outputs = encoder_outputs[::-1]
        x = self.up1(encoder_outputs[0], encoder_outputs[1])
        x = self.up2(x, encoder_outputs[2])
        x = self.up3(x, encoder_outputs[3])
        x = self.up4(x, encoder_outputs[4])
        x = self.outc(x)
        return x


class Unet(nn.Module):
    def __init__(self, in_channels):
        super(Unet, self).__init__()
        self.n_channels = in_channels
        self.encoder = Encoder
        self.decoder = Decoder(in_channels=in_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


def get_unet_transformations():
    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(config.RESIZE),
            torchvision.transforms.CenterCrop(config.CROP),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=config.MEAN, std=config.STD),
        ]
    )

    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(config.RESIZE),
            torchvision.transforms.CenterCrop(config.CROP),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=config.MEAN, std=config.STD),
        ]
    )

    return train_transform, test_transform
