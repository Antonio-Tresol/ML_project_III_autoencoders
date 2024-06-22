# Imports

# System
import os

# Utils
import pickle
import numpy as np
from typing import List, Union
from enum import Enum

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# Torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from pytorch_lightning import LightningDataModule

# Customs
from data.datasets import (
    ImageAutoencoderFolderDataset,
    ImageClassificationFolderDataset,
)
import configuration as config


# Enum for different sampling strategies
class Sampling(Enum):
    NUMPY = 1
    SKLEARN = 2
    NONE = 3


# Utility class for managing indices
class IndexManager:
    @staticmethod
    def save_indices(indices: list[int], indices_path: str):
        """
        Save indices to a file.

        Args:
            indices (tuple): Tuple containing train and test indices.
            indices_path (str): Path to the file where indices will be saved.
        """
        with open(indices_path, "wb") as file:
            pickle.dump(indices, file)

    @staticmethod
    def load_indices(indices_path: str):
        """
        Load indices from a file.

        Args:
            indices_path (str): Path to the file containing saved indices.

        Returns:
            tuple: Tuple containing train and test indices.
        """
        with open(indices_path, "rb") as file:
            return pickle.load(file)
    @staticmethod    
    def create_indices(folder_dataset, indices_path, train_size, test_size, preset_indices):
        """
        Create indices for training and testing.

        Args:
            folder_dataset (Dataset): Dataset instance.
            indices_path (str): Path to the file where indices will be saved.
            train_size (float): Fraction of the data to reserve as training set.
            test_size (float): Fraction of the data to reserve as test set.
            use_index (bool): Flag indicating whether to use existing indices.
            preset_indices (list, optional): List of indices to use for splitting data. Default is None.

        Returns:
            tuple: Tuple containing train and test indices.
        """
        if preset_indices is not None:
            indices = train_test_split(
                preset_indices,
                train_size=train_size,
                test_size=test_size,
                stratify=folder_dataset.labels,
            )
        else:
            indices = train_test_split(
                range(len(folder_dataset)),
                train_size=train_size,
                test_size=test_size,
                stratify=folder_dataset.labels,
            )
        IndexManager.save_indices(indices, indices_path)
        return indices


# Utility class for splitting data into train and test sets
class DataSplitter:
    @staticmethod
    def split_data(
        folder_dataset: ImageClassificationFolderDataset,
        indices_path: str,
        train_size: float,
        test_size: float,
        use_index: bool,
        preset_indices: List[int] = None,
    ):
        """
        Split data into train and test indices.

        Args:
            folder_dataset (Dataset): Dataset instance.
            indices_path (str): Path to the file where indices will be saved.
            test_size (float): Fraction of the data to reserve as test set.
            use_index (bool): Flag indicating whether to use existing indices.
            preset_indices (list, optional): List of indices to use for splitting data. Default is None.

        Returns:
            tuple: Tuple containing train and test indices.
        """
        if use_index:
            try:
                indices = IndexManager.load_indices(indices_path)
            except:
                indices = IndexManager.create_indices(folder_dataset, indices_path, train_size, test_size, preset_indices)
            return indices
        else:
            indices = IndexManager.create_indices(folder_dataset, indices_path, train_size, test_size, preset_indices)
            return indices


# Utility class for creating data loaders
class DataLoaderCreator:
    @staticmethod
    def create_dataloader(
        dataset: Dataset,
        batch_size: int,
        sampler=None,
        shuffle: bool = False,
        num_workers: int = 1,
    ):
        """
        Create a DataLoader for a dataset.

        Args:
            dataset (Dataset): Dataset instance.
            sampler (optional): Sampler used for sampling data. Default is None.
            shuffle (bool, optional): Flag indicating whether to shuffle the data. Default is False.
            num_workers (int, optional): Number of subprocesses to use for data loading. Default is 1.

        Returns:
            DataLoader: DataLoader instance.
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        )


class SamplerFactory:
    @staticmethod
    def create_sampler(sampling: Sampling, train_dataset: Dataset, train_labels):
        """
        Create a sampler based on the specified sampling strategy.

        Args:
            sampling (Sampling): Enum value indicating the sampling strategy.
            train_dataset (Dataset): Training dataset.
            train_labels (optional): Train labels.

        Returns:
            Sampler or None: Sampler instance based on the specified strategy, or None if no sampler is needed.
        """
        if sampling == Sampling.NONE:
            return None

        elif sampling == Sampling.NUMPY:
            class_counts = np.array(
                [np.sum(train_labels == c) for c in np.unique(train_labels)]
            )
            class_weights = 1 / class_counts

            return WeightedRandomSampler(class_weights, len(train_dataset))
        else:
            class_weights = class_weight.compute_class_weight(
                class_weight="balanced", classes=np.unique(train_labels), y=train_labels
            )
            return WeightedRandomSampler(class_weights, len(train_dataset))


class ImagesDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: str,
        root_dir: str,
        batch_size: int,
        train_folder_dataset: Union[
            ImageClassificationFolderDataset, ImageAutoencoderFolderDataset
        ],
        test_folder_dataset: Union[
            ImageClassificationFolderDataset, ImageAutoencoderFolderDataset
        ],
        train_size: float = 0.5,
        test_size: float = 0.5,
        use_index: bool = True,
        indices_dir: str = None,
        preset_indices: List[int] = None,
        sampling: Sampling = Sampling.NONE,
    ):
        """
        Initialize the ImageDataModule.

        Args:
            dataset (str): Name of the dataset.
            root_dir (str): Root directory of the dataset.
            batch_size (int): Batch size for data loaders.
            train_folder_dataset (Dataset): Dataset class to use for training.
            test_folder_dataset (Dataset): Dataset class to use for testing.
            train_size (float, optional): Fraction of data to use as training set. Default is 0.5.
            test_size (float, optional): Fraction of data to use as test set. Default is 0.5.
            use_index (bool, optional): Whether to use existing indices. Default is True.
            indices_dir (str, optional): Directory to save indices. Default is None.
            preset_indices (list, optional): List of indices to use for splitting data. Default is None.
            sampling (Sampling, optional): Sampling strategy. Default is Sampling.NONE.
        """
        super().__init__()
        self.save_hyperparameters()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.train_size = train_size
        self.test_size = test_size
        self.use_index = use_index
        self.sampling = sampling

        # Initialize training and test folders
        self.train_folder = train_folder_dataset
        self.test_folder = test_folder_dataset

        self.class_counts = self.train_folder.class_counts
        self.classes = self.train_folder.classes
        self.indices_path = os.path.join(indices_dir, str(dataset) + ".pkl")
        self.preset_indices = preset_indices

    def prepare_data(self):
        """
        Prepare data for training and testing.
        """
        # Split train and test indices
        self.train_indices, self.test_indices = DataSplitter.split_data(
            self.train_folder, self.indices_path, self.train_size, self.test_size, self.use_index, self.preset_indices
        )
        # Split the datasets
        self.train_dataset = Subset(self.train_folder, self.train_indices)
        self.test_dataset = Subset(self.test_folder, self.test_indices)
        train_labels = np.array(self.train_folder.labels)[self.train_indices]
        # Create a sampler (if needed)
        self.train_sampler = SamplerFactory.create_sampler(
            self.sampling, self.train_dataset, train_labels
        )

    def create_data_loaders(self):
        """
        Create data loaders for training and testing.
        """
        # Shuffle flag
        shuffle = True if self.sampling == Sampling.NONE else False
        # Create data loaders
        self.train_loader = DataLoaderCreator.create_dataloader(
            self.train_dataset,
            self.batch_size,
            self.train_sampler,
            shuffle=shuffle,
            num_workers=8,
        )
        self.test_loader = DataLoaderCreator.create_dataloader(
            self.test_dataset, self.batch_size, num_workers=8
        )

    def train_dataloader(self):
        """
        Get the training data loader.

        Returns:
            DataLoader: Training data loader.
        """
        return self.train_loader

    def val_dataloader(self):
        """
        Get the validation data loader (same as test data loader).

        Returns:
            DataLoader: Validation data loader.
        """
        return self.test_loader

    def test_dataloader(self):
        """
        Get the test data loader.

        Returns:
            DataLoader: Test data loader.
        """
        return self.test_loader

