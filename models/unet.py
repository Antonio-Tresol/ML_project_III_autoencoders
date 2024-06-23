import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import configuration as config

class DoubleConv(nn.Module):
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


class InConv(nn.Module):
    """
    Initial convolution module in a U-Net architecture.

    This module applies a double convolution to the input tensor.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    output_channels : int
        Number of output channels.

    Attributes
    ----------
    conv : DoubleConv
        The double convolution module.
    """

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv = DoubleConv(input_channels, output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the InConv module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor after applying the double convolution.
        """
        x = self.conv(x)
        return x


class Down(nn.Module):
    """
    Downscaling module with max pooling and double convolution.

    This module applies a max pooling operation followed by a double convolution.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    output_channels : int
        Number of output channels.
    stride : int, optional
        Stride of the max pooling operation. Default is 1.

    Attributes
    ----------
    mpconv : nn.Sequential
        Sequential container of max pooling and double convolution.
    """

    def __init__(
        self, input_channels: int, output_channels: int, stride: int = 1
    ) -> None:
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2, stride=stride), DoubleConv(input_channels, output_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Down module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor after max pooling and double convolution.
        """
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    """
    Upsampling module with a convolution transpose operation followed by a double convolution.

    This module performs upsampling on the input tensor and merges it with a skip connection
    before applying a double convolution.

    Parameters
    ----------
    input_channels : int
        Number of input channels for the upsampling module.
    output_channels : int
        Number of output channels after the double convolution.

    Attributes
    ----------
    up : nn.ConvTranspose2d
        Convolution transpose layer for upsampling.
    conv : DoubleConv
        Double convolution module to process the merged tensor.
    """

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(
            input_channels // 2, input_channels // 2, 2, stride=2
        )
        self.conv = DoubleConv(input_channels, output_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Up module.

        Parameters
        ----------
        x1 : torch.Tensor
            The tensor to be upsampled.
        x2 : torch.Tensor
            The tensor from the skip connection to be merged with x1.

        Returns
        -------
        torch.Tensor
            The output tensor after upsampling, merging, and double convolution.
        """
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    """
    Output convolution module.

    This module applies a single convolution to the input tensor to produce the output tensor
    with the desired number of channels.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    output_channels : int
        Number of output channels.

    Attributes
    ----------
    conv : nn.Conv2d
        Convolution layer to produce the output tensor.
    """

    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the OutConv module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor with the specified number of channels.
        """
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    """
    Encoder module for a U-Net architecture.

    This module sequentially applies four downsampling operations to the input tensor,
    each consisting of a max pooling operation followed by a double convolution.

    Parameters
    ----------
    input_channels : int
        Number of channels in the input tensor.

    Attributes
    ----------
    n_channels : int
        Stores the number of input channels.
    inc : InConv
        Initial convolution module.
    down1 : Down
        First downsampling module.
    down2 : Down
        Second downsampling module.
    down3 : Down
        Third downsampling module.
    down4 : Down
        Fourth downsampling module.
    """

    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.n_channels = input_channels
        self.inc = InConv(input_channels, 64)
        self.down1 = Down(64, 128, stride=4)
        self.down2 = Down(128, 256, stride=4)
        self.down3 = Down(256, 512, stride=4)
        self.down4 = Down(512, 512, stride=4)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass of the Encoder module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        List[torch.Tensor]
            A list of tensors corresponding to the output of each downsampling stage and the initial convolution.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        outputs = [x1, x2, x3, x4, x5]
        return outputs


class Decoder(nn.Module):
    """
    Decoder module for a U-Net architecture.

    This module sequentially applies four upsampling operations to the set of encoder outputs,
    each consisting of an upsampling operation followed by a double convolution. The final
    layer is a convolution that maps to the desired number of output channels.

    Parameters
    ----------
    output_channels : int
        Number of channels in the output tensor.

    Attributes
    ----------
    up1 : Up
        First upsampling module.
    up2 : Up
        Second upsampling module.
    up3 : Up
        Third upsampling module.
    up4 : Up
        Fourth upsampling module.
    outc : OutConv
        Output convolution module.
    """

    def __init__(self, output_channels: int) -> None:
        super().__init__()
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, output_channels)

    def forward(self, encoder_outputs: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the Decoder module.

        Parameters
        ----------
        encoder_outputs : List[torch.Tensor]
            A list of tensors from the encoder module.

        Returns
        -------
        torch.Tensor
            The output tensor after all upsampling operations and the final convolution.
        """
        encoder_outputs = encoder_outputs[::-1]
        x = self.up1(encoder_outputs[0], encoder_outputs[1])
        x = self.up2(x, encoder_outputs[2])
        x = self.up3(x, encoder_outputs[3])
        x = self.up4(x, encoder_outputs[4])
        x = self.outc(x)
        return x


class Unet(nn.Module):
    """
    U-Net architecture for image segmentation.

    This class implements the U-Net architecture, which consists of an encoder module for downsampling the input,
    and a decoder module for upsampling the feature maps to the original image size. The U-Net architecture is
    widely used for biomedical image segmentation tasks.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    device : torch.device
        The device (CPU or GPU) the model should be allocated to.

    Attributes
    ----------
    n_channels : int
        Stores the number of input channels.
    encoder : Encoder
        The encoder module of the U-Net.
    decoder : Decoder
        The decoder module of the U-Net.
    """

    def __init__(self, in_channels: int, device: torch.device) -> None:
        super().__init__()
        self.n_channels = in_channels
        self.encoder = Encoder(input_channels=in_channels).to(device)
        self.decoder = Decoder(output_channels=in_channels).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the U-Net model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor after passing through the encoder and decoder modules.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def get_unet_transformations():
    """
    Defines the transformations for training and testing datasets for U-Net.

    This function creates and returns the transformations needed to preprocess the images
    for training and testing the U-Net model. The transformations include resizing, center cropping,
    converting to tensor, and normalizing with predefined mean and standard deviation values.

    Returns
    -------
    tuple
        A tuple containing the training and testing transformations.
    """
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
