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

    from models.lightning_module import AutoencoderLightningModule, ClassificationLightningModule
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping
    from data.datasets import ImageClassificationFolderDataset
    from data.data_modules import ImagesDataModule, Sampling, IndexManager
    from torchmetrics.classification import (
        MulticlassAccuracy,
        MulticlassPrecision,
        MulticlassRecall,
    )
    from torchmetrics import MetricCollection
    from models.unet import Unet, get_unet_transformations
    from models.mlp import MLP
    from models.autoencoder_classifier import AutoencoderClassifier
    from torch import nn
    import wandb
    import configuration as config

    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class_count = count_classes(config.ROOT_DIR)

    metrics = MetricCollection(
        {
            "Accuracy": MulticlassAccuracy(num_classes=class_count, average="micro"),
            "Precision": MulticlassPrecision(num_classes=class_count),
            "Recall": MulticlassRecall(num_classes=class_count),
        }
    )
    vector_metrics = MetricCollection(
        {
            "Accuracy": MulticlassAccuracy(num_classes=class_count, average=None),
            "Precision": MulticlassPrecision(num_classes=class_count, average=None),
            "Recall": MulticlassRecall(num_classes=class_count, average=None),
        }
    )

    train_transform, test_transform = get_unet_transformations()

    train_dataset = ImageClassificationFolderDataset(
        root_dir=config.ROOT_DIR, transform=train_transform
    )
    test_dataset = ImageClassificationFolderDataset(
        root_dir=config.ROOT_DIR, transform=test_transform
    )

    preset_indices = IndexManager.load_indices(os.path.join(config.INDICES_DIR, config.DATASET_80_20_NAME + ".pkl"))[1]
    
    plant_dm = ImagesDataModule(
        dataset=config.DATASET_80_10_10_NAME,
        root_dir=config.ROOT_DIR,
        batch_size=config.BATCH_SIZE,
        train_folder_dataset=train_dataset,
        test_folder_dataset=test_dataset,
        train_size=config.TRAIN_SIZE_80_10_10,
        test_size=config.TEST_SIZE_80_10_10,
        use_index=config.USE_INDEX,
        indices_dir=config.INDICES_DIR,
        preset_indices=preset_indices,
        sampling=Sampling.NONE,
    )

    plant_dm.prepare_data()
    plant_dm.create_data_loaders()

    for i in range(config.NUM_TRIALS):
        checkpoint_filename = (
            config.AUTOENCODER_80_20_DIR + config.AUTOENCODER_80_20_FILENAME + str(i) + ".ckpt"
        )
        unet = Unet(in_channels=3, device=device)
        encoder = AutoencoderLightningModule.load_from_checkpoint(
            checkpoint_path=checkpoint_filename,
            model=unet,
            model_name=config.AUTOENCODER_80_20_FILENAME.replace("_", ""),
            loss_fn=nn.MSELoss(),
            metrics=metrics,
            lr=config.LR,
            scheduler_max_it=config.SCHEDULER_MAX_IT,
            class_names=config.CLASS_NAMES,
        ).model.encoder

        classifier = MLP(input_size=config.MLP_INPUT_SIZE, hidden_layer_count=config.MLP_HIDDEN_LAYERS, hidden_layer_size=config.MLP_HIDDEN_DIM, output_size=class_count, device=device)
        autoencoder_classifier = AutoencoderClassifier(encoder, classifier, freeze_encoder=True, device=device)
        
        model = ClassificationLightningModule(
            model=autoencoder_classifier,
            model_name=config.FREEZED_AUTOENCODER_CLASSIFIER_80_10_10_FILENAME.replace("_", ""),
            loss_fn=nn.CrossEntropyLoss(),
            metrics=metrics,
            lr=config.LR,
            scheduler_max_it=config.SCHEDULER_MAX_IT,
            per_class_metrics=vector_metrics,
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
            dirpath=config.FREEZED_AUTOENCODER_CLASSIFIER_80_10_10_DIR,
            filename=config.FREEZED_AUTOENCODER_CLASSIFIER_80_10_10_FILENAME + str(i),
            save_top_k=config.TOP_K_SAVES,
            mode="min",
        )

        id = (
            config.FREEZED_AUTOENCODER_CLASSIFIER_80_10_10_FILENAME + str(i) + "_" + wandb.util.generate_id()
        )
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
