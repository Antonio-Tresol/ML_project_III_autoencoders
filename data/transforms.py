import torch

class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        """
        Add Gaussian noise to a tensor.
        
        Args:
            mean (float): The mean of the Gaussian distribution.
            std (float): The standard deviation of the Gaussian distribution.
        """
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor: torch.Tensor):
        """
        Add Gaussian noise to the tensor.
        
        Args:
            tensor (torch.Tensor): The input tensor.
            
        Returns:
            torch.Tensor: The tensor with added Gaussian noise.
        """
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        """
        Get the string representation of the GaussianNoise object.
        
        Returns:
            str: The string representation of the GaussianNoise object.
        """
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)