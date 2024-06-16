import torch
import torch.nn as nn
import torchvision
import configuration as config
from typing import Optional

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.model import UnetDecoder

class Unet(nn.Module):
    """Unet_ is a fully convolution neural network for image semantic segmentation
    
    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_weights: one of ``None
            (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_use_batchnorm: if ``True``, batch normalization layer between encoder and decoder
        in the segmentation model is used.
        decoder_channels: number of convolution filters in each decoder block.
        decoder_attention_type: attention module used in decoder of the segmentation model.
        in_channels: number of input channels for the model (default 3).
        
    Returns:
        ``torch.nn.Module``: **Unet**
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: list[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        device = "cuda"
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        ).to(device)

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        ).to(device)

        self.name = "u-{}".format(encoder_name)
    
    def forward(self, x):
        """Sequentially pass `x` through model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        return decoder_output
    
def get_unet_transformations():
    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(config.RESIZE),
            torchvision.transforms.CenterCrop(config.CROP),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=config.MEAN, std=config.STD)
        ]
    )

    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(config.RESIZE),
            torchvision.transforms.CenterCrop(config.CROP),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=config.MEAN, std=config.STD)
        ]
    )

    return train_transform, test_transform