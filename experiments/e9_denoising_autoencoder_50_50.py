def main():
    import os
    import sys
    import inspect

    currentdir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)

    import pandas as pd
    import torch
    from pytorch_lightning.loggers import WandbLogger
    from helper_functions import count_classes

    from models.lightning_module import AutoencoderLightningModule
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping
    from data.datasets import ImageDenoisingAutoencoderFolderDataset
    from data.data_modules import ImagesDataModule, Sampling
    from data.transforms import GaussianNoise
    from torchmetrics.regression import MeanAbsoluteError

    from torchmetrics import MetricCollection
    from models.unet import Unet, get_unet_transformations
    from torch import nn
    import wandb
    import configuration as config

    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class_count = count_classes(config.ROOT_DIR)

    metrics = MetricCollection(
        {
            "AbsoulteError": MeanAbsoluteError(),
        }
    )

    train_transform, test_transform = get_unet_transformations()

    train_dataset = ImageDenoisingAutoencoderFolderDataset(
        root_dir=config.ROOT_DIR, noise_transform=GaussianNoise(), transform=train_transform
    )
    test_dataset = ImageDenoisingAutoencoderFolderDataset(
        root_dir=config.ROOT_DIR, noise_transform=GaussianNoise(), transform=test_transform
    )

    plant_dm = ImagesDataModule(
        dataset=config.DATASET_50_50_NAME,
        root_dir=config.ROOT_DIR,
        batch_size=config.BATCH_SIZE,
        train_folder_dataset=train_dataset,
        test_folder_dataset=test_dataset,
        train_size=config.TRAIN_SIZE_50_50,
        test_size=config.TEST_SIZE_50_50,
        use_index=config.USE_INDEX,
        indices_dir=config.INDICES_DIR,
        sampling=Sampling.NONE,
    )

    plant_dm.prepare_data()
    plant_dm.create_data_loaders()

    for i in range(config.NUM_TRIALS):
        unet = Unet(in_channels=3, device=device)
        model = AutoencoderLightningModule(
            model=unet,
            model_name=config.DENOISING_AUTOENCODER_50_50_FILENAME.replace("_", ""),
            loss_fn=nn.MSELoss(),
            metrics=metrics,
            lr=config.LR,
            scheduler_max_it=config.SCHEDULER_MAX_IT,
            class_names=config.CLASS_NAMES,
        )

        early_stop_callback = EarlyStopping(
            monitor="val/loss",
            patience=config.PATIENCE,
            strict=False,
            verbose=False,
            mode="min",
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val/loss",
            dirpath=config.DENOISING_AUTOENCODER_50_50_DIR,
            filename=config.DENOISING_AUTOENCODER_50_50_FILENAME + str(i),
            save_top_k=config.TOP_K_SAVES,
            mode="min",
        )

        id = config.DENOISING_AUTOENCODER_50_50_FILENAME + str(i) + "_" + wandb.util.generate_id()
        wandb_logger = WandbLogger(project=config.WANDB_PROJECT, id=id, resume="allow")

        trainer = Trainer(
            logger=wandb_logger,
            callbacks=[early_stop_callback, checkpoint_callback],
            max_epochs=config.EPOCHS,
            log_every_n_steps=1,
        )

        trainer.fit(model, datamodule=plant_dm)
        trainer.test(model, datamodule=plant_dm)

        wandb.finish()


if __name__ == "__main__":
    main()
