from torch import nn
import torch
import torchvision
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import configuration as conf


class AutoencoderClassifier(nn.Module):
    def __init__(self, encoder, classifier, freeze_encoder, device) -> None:
        """
        Initializes a AutoencoderClassifier model.

        Args:
            num_classes (int): The number of output classes.
            device (torch.device): The device to run the model on.
        """
        super().__init__()
        self.encoder = encoder.to(device)
        self.flatten = nn.Flatten().to(device)
        self.classifier = classifier.to(device)
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the ConvNext model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.encoder(x)
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[-1]
        
        x = self.flatten(x)
        return self.classifier(x)

def get_conv_model_transformations() -> tuple[torchvision.transforms.Compose]:
    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(conf.RESIZE),
            torchvision.transforms.CenterCrop(conf.CROP),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomRotation(conf.ROTATION),
            torchvision.transforms.Normalize(conf.MEAN, conf.STD),
        ]
    )

    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(conf.RESIZE),
            torchvision.transforms.CenterCrop(conf.CROP),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(conf.MEAN, conf.STD),
        ]
    )
    return train_transform, test_transform
