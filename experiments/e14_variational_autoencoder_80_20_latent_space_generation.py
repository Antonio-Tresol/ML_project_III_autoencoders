def main():
    import os
    import sys
    import inspect

    currentdir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)

    import torch
    from pytorch_lightning.loggers import WandbLogger
    from helper_functions import count_classes

    from models.lightning_module import VariationalAutoencoderLightningModule
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping
    from data.datasets import ImageAutoencoderFolderDataset
    from data.data_modules import ImagesDataModule, Sampling
    from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

    from torchmetrics import MetricCollection
    from models.unet import get_unet_transformations
    from models.vae import VAE, vae_loss_fn
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

    train_dataset = ImageAutoencoderFolderDataset(
        root_dir=config.ROOT_DIR, transform=train_transform
    )
    test_dataset = ImageAutoencoderFolderDataset(
        root_dir=config.ROOT_DIR, transform=test_transform
    )

    plant_dm = ImagesDataModule(
        dataset=config.DATASET_80_20_NAME,
        root_dir=config.ROOT_DIR,
        batch_size=config.BATCH_SIZE,
        train_folder_dataset=train_dataset,
        test_folder_dataset=test_dataset,
        train_size=config.TRAIN_SIZE_80_20,
        test_size=config.TEST_SIZE_80_20,
        use_index=config.USE_INDEX,
        indices_dir=config.INDICES_DIR,
        sampling=Sampling.NONE,
    )

    plant_dm.prepare_data()
    plant_dm.create_data_loaders()

    for i in range(config.NUM_TRIALS):
        vae_unet = VAE().to(device)

        model = VariationalAutoencoderLightningModule.load_from_checkpoint(
            config.VAE_80_20_DIR + config.VAE_80_20_FILENAME + "0.ckpt",
            model=vae_unet,
            model_name=config.VAE_80_20_FILENAME.replace("_", ""),
            loss_fn=vae_loss_fn,
            metrics=metrics,
            lr=config.LR,
            scheduler_max_it=config.SCHEDULER_MAX_IT,
            class_names=config.CLASS_NAMES,
        )

        id = config.VAE_80_20_FILENAME + str(i) + "_" + wandb.util.generate_id()
        wandb_logger = WandbLogger(project=config.WANDB_PROJECT, id=id, resume="allow")

        trainer = Trainer(
            logger=wandb_logger,
            max_epochs=config.EPOCHS,
            log_every_n_steps=1,
        )
        trainer.test(model, datamodule=plant_dm)

        wandb.finish()


if __name__ == "__main__":
    main()
