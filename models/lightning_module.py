import torch
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from abc import ABC, abstractmethod
from models.logger import *
from models.metrics_manager import *
import pandas as pd
import wandb
import configuration as config
import numpy as np
from itertools import chain


class BaseLightningModule(LightningModule):
    """
    LightningModule Base Module.

    Args:
        model: Model instance representing the Network Model.
        model_name: Model's name.
        loss_fn: Loss function used for training.
        metrics: Metrics used for evaluation.
        lr (float): Learning rate for the optimizer.
        scheduler_max_it (int): Maximum number of iterations for the learning rate scheduler.
        weight_decay (float): Weight decay for the optimizer.

    Attributes:
        model: Model instance representing the Convolution Network Model.
        model_name: Model's name.
        loss_fn: Loss function used for training.
        train_metrics: Metrics used for training evaluation.
        val_metrics: Metrics used for validation evaluation.
        test_metrics: Metrics used for testing evaluation.
        lr (float): Learning rate for the optimizer.
        scheduler_max_it (int): Maximum number of iterations for the learning rate scheduler.
    """

    def __init__(
        self, model, model_name, loss_fn, metrics, lr, scheduler_max_it, weight_decay=0
    ):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.loss_fn = loss_fn
        self.scheduler_max_it = scheduler_max_it
        self.weight_decay = weight_decay
        self.lr = lr

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.train_metrics_manager: BaseMetricsManager
        self.val_metrics_manager: BaseMetricsManager
        self.test_metrics_manager: BaseMetricsManager

        self.train_loss_logger: BaseMetricsLogger
        self.val_loss_logger: BaseMetricsLogger
        self.test_loss_logger: BaseMetricsLogger

        self.train_metrics_logger: BaseMetricsLogger
        self.val_metrics_logger: BaseMetricsLogger
        self.test_metrics_logger: BaseMetricsLogger

    @abstractmethod
    def configure_metrics_managers(self):
        """
        Configure metrics managers.
        """
        pass

    @abstractmethod
    def configure_loggers(self):
        """
        Configure loggers.
        """
        pass

    def forward(self, X):
        """
        Forward pass of the Convolutional model.

        Args:
            X: Input tensor.

        Returns:
            Output tensor.
        """
        outputs = self.model(X)
        return outputs

    def _final_step(self, y_hat):
        """
        Final step of the forward pass. Includes the final activation function if any and the final prediction.
        This should be overriden by the subclass if needed (for example for classification tasks).

        Args:
            y_hat: Predicted outputs.

        Returns:
            Predicted outputs.
        """
        return y_hat

    def _common_step(self, batch, batch_idx):
        """
        Common step for training, validation, and testing.

        Args:
            batch: Input batch.
            batch_idx: Index of the current batch.

        Returns:
            Tuple containing the ground truth labels, predicted outputs, and loss value.
        """
        x, y = batch
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)
        return y, y_hat, loss

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch: Input batch.
            batch_idx: Index of the current batch.

        Returns:
            Dictionary containing the loss value, ground truth labels, and predicted outputs.
        """
        y, y_hat, loss = self._common_step(batch, batch_idx)
        y_hat = self._final_step(y_hat)

        self.train_metrics_manager.update_metrics(y_true=y, y_pred=y_hat)

        self.train_loss_logger.log(loss)
        return {"loss": loss, "train/labels": y, "train/predictions": y_hat}

    def on_train_epoch_end(self):
        """
        Callback function called at the end of each training epoch.
        Computes and logs the training metrics.
        """
        self.train_metrics_logger.log()
        self.train_metrics_manager.reset_metrics()

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch: Input batch.
            batch_idx: Index of the current batch.

        Returns:
            Dictionary containing the loss value, ground truth labels, and predicted outputs.
        """
        y, y_hat, loss = self._common_step(batch, batch_idx)
        y_hat = self._final_step(y_hat)

        self.val_metrics_manager.update_metrics(y_true=y, y_pred=y_hat)

        self.val_loss_logger.log(loss)
        return {"loss": loss, "val/labels": y, "val/predictions": y_hat}

    def on_validation_epoch_end(self):
        """
        Callback function called at the end of each validation epoch.
        Computes and logs the validation metrics.
        """
        self.val_metrics_logger.log()
        self.val_metrics_manager.reset_metrics()

    def test_step(self, batch, batch_idx):
        """
        Test step.

        Args:
            batch: Input batch.
            batch_idx: Index of the current batch.

        Returns:
            Dictionary containing the loss value, ground truth labels, and predicted outputs.
        """
        y, y_hat, loss = self._common_step(batch, batch_idx)
        y_hat = self._final_step(y_hat)

        self.test_metrics_manager.update_metrics(y_true=y, y_pred=y_hat)

        self.test_loss_logger.log(loss)
        return {"loss": loss, "test/labels": y, "test/predictions": y_hat}

    def on_test_epoch_end(self):
        """
        Callback function called at the end of each testing epoch.
        Computes and logs the testing metrics.
        """
        self.test_metrics_logger.log()
        self.test_metrics_manager.reset_metrics()

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            Tuple containing the optimizer and learning rate scheduler.
        """
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.scheduler_max_it)
        return [optimizer], [scheduler]


class ClassificationLightningModule(BaseLightningModule):
    """
    LightningModule implementation for classification tasks.

    Args:
        model: Model instance representing the Network Model.
        model_name: Model's name.
        class_names: List of class names.
        loss_fn: Loss function used for training.
        metrics: Metrics used for evaluation.
        lr (float): Learning rate for the optimizer.
        scheduler_max_it (int): Maximum number of iterations for the learning rate scheduler.
        weight_decay (float): Weight decay for the optimizer.
        per_class_metrics: Metrics that return a vector as a result of the computing.

    Attributes:
        model: Model instance representing the Convolution Network Model.
        model_name: Model's name.
        class_names: List of class names.
        loss_fn: Loss function used for training.
        train_metrics: Metrics used for training evaluation.
        val_metrics: Metrics used for validation evaluation.
        test_metrics: Metrics used for testing evaluation.
        lr (float): Learning rate for the optimizer.
        scheduler_max_it (int): Maximum number of iterations for the learning rate scheduler.
    """

    def __init__(
        self,
        model,
        model_name,
        class_names,
        loss_fn,
        metrics,
        lr,
        scheduler_max_it,
        weight_decay=0,
        per_class_metrics=None,
    ):
        super(ClassificationLightningModule, self).__init__(
            model=model,
            model_name=model_name,
            loss_fn=loss_fn,
            metrics=metrics,
            lr=lr,
            scheduler_max_it=scheduler_max_it,
            weight_decay=weight_decay,
        )

        self.class_names = class_names
        self.per_class_metrics = per_class_metrics
        self.test_all_preds = []
        self.test_all_targets = []

        self.configure_metrics_managers()
        self.configure_loggers()

    def _final_step(self, y_hat):
        """
        Final step of the forward pass. Includes the final activation function if any and the final prediction.
        This should be overriden by the subclass if needed (for example for classification tasks).

        Args:
            y_hat: Predicted outputs.

        Returns:
            Predicted outputs.
        """
        return torch.argmax(torch.softmax(y_hat, dim=-1), dim=-1)

    def configure_metrics_managers(self):
        """
        Configure metrics managers.
        """
        self.train_metrics_manager = MetricsManager(
            module=self, metrics=self.train_metrics
        )
        self.val_metrics_manager = MetricsManager(module=self, metrics=self.val_metrics)

        if self.per_class_metrics is not None:
            self.test_metrics_manager = MetricsManagerCollection(
                module=self,
                managers=[
                    MetricsManager(module=self, metrics=self.test_metrics),
                    MetricsManager(module=self, metrics=self.per_class_metrics),
                ],
            )
        else:
            self.test_metrics_manager = MetricsManager(
                module=self, metrics=self.test_metrics
            )

    def configure_loggers(self):
        """
        Configure loggers.
        """
        self.train_loss_logger = ScalarLogger(
            prefix="train/", module=self, scalar_name="loss"
        )
        self.val_loss_logger = ScalarLogger(
            prefix="val/", module=self, scalar_name="loss"
        )
        self.test_loss_logger = ScalarLogger(
            prefix="test/", module=self, scalar_name="loss"
        )

        self.train_metrics_logger = DictLogger(
            prefix="train/", module=self, metrics=self.train_metrics
        )
        self.val_metrics_logger = DictLogger(
            prefix="val/", module=self, metrics=self.val_metrics
        )

        if self.per_class_metrics is not None:
            self.test_metrics_logger = LoggerCollection(
                prefix="test/",
                module=self,
                loggers=[
                    DictLogger(prefix="test/", module=self, metrics=self.test_metrics),
                    MetricsTableLogger(
                        prefix="test/",
                        module=self,
                        metrics=self.test_metrics,
                        table_name="Metrics",
                    ),
                    PerClassLogger(
                        prefix="test/", module=self, metrics=self.per_class_metrics
                    ),
                    ConfusionMatrixLogger(
                        prefix="test/",
                        module=self,
                        y_true_ref=self.test_all_targets,
                        y_pred_ref=self.test_all_preds,
                    ),
                ],
            )
        else:
            self.test_metrics_logger = LoggerCollection(
                prefix="test/",
                module=self,
                loggers=[
                    DictLogger(prefix="test/", module=self, metrics=self.test_metrics),
                    MetricsTableLogger(
                        prefix="test/",
                        module=self,
                        metrics=self.test_metrics,
                        table_name="Metrics",
                    ),
                    ConfusionMatrixLogger(prefix="test/", module=self),
                ],
            )

    def test_step(self, batch, batch_idx):
        """
        Test step.

        Args:
            batch: Input batch.
            batch_idx: Index of the current batch.

        Returns:
            Dictionary containing the loss value, ground truth labels, and predicted outputs.
        """
        test_results = super().test_step(batch, batch_idx)

        self.test_all_preds.append(test_results["test/predictions"])
        self.test_all_targets.append(test_results["test/labels"])

        return test_results

    def on_test_epoch_end(self):
        """
        Callback function called at the end of each testing epoch.
        Computes and logs the testing metrics.
        """
        super().on_test_epoch_end()

        self.test_all_preds = []
        self.test_all_targets = []


class AutoencoderLightningModule(BaseLightningModule):
    """
    LightningModule Autoencoder Module.

    Args:
        model: Model instance representing the Network Model.
        model_name: Model's name.
        loss_fn: Loss function used for training.
        metrics: Metrics used for evaluation.
        lr (float): Learning rate for the optimizer.
        scheduler_max_it (int): Maximum number of iterations for the learning rate scheduler.
        weight_decay (float): Weight decay for the optimizer.

    Attributes:
        model: Model instance representing the Convolution Network Model.
        model_name: Model's name.
        loss_fn: Loss function used for training.
        train_metrics: Metrics used for training evaluation.
        val_metrics: Metrics used for validation evaluation.
        test_metrics: Metrics used for testing evaluation.
        lr (float): Learning rate for the optimizer.
        scheduler_max_it (int): Maximum number of iterations for the learning rate scheduler.
    """

    def __init__(
        self,
        model,
        model_name,
        loss_fn,
        metrics,
        lr,
        scheduler_max_it,
        weight_decay=0,
        class_names=None,
    ):
        super(AutoencoderLightningModule, self).__init__(
            model=model,
            model_name=model_name,
            loss_fn=loss_fn,
            metrics=metrics,
            lr=lr,
            scheduler_max_it=scheduler_max_it,
            weight_decay=weight_decay,
        )
        self.class_names = class_names
        self.test_latent_vectors = []
        self.test_labels = []
        self.test_images = []
        self.test_latent_space = []

        self.configure_metrics_managers()
        self.configure_loggers()

    def configure_metrics_managers(self):
        """
        Configure metrics managers.
        """
        self.train_metrics_manager = MetricsManager(
            module=self, metrics=self.train_metrics
        )
        self.val_metrics_manager = MetricsManager(module=self, metrics=self.val_metrics)
        self.test_metrics_manager = MetricsManager(
            module=self, metrics=self.test_metrics
        )

    def configure_loggers(self):
        """
        Configure loggers.
        """
        self.train_loss_logger = ScalarLogger(
            prefix="train/", module=self, scalar_name="loss"
        )
        self.val_loss_logger = ScalarLogger(
            prefix="val/", module=self, scalar_name="loss"
        )
        self.test_loss_logger = ScalarLogger(
            prefix="test/", module=self, scalar_name="loss"
        )

        self.train_metrics_logger = DictLogger(
            prefix="train/", module=self, metrics=self.train_metrics
        )
        self.val_metrics_logger = DictLogger(
            prefix="val/", module=self, metrics=self.val_metrics
        )
        self.test_metrics_logger = self.test_metrics_logger = LoggerCollection(
            prefix="test/",
            module=self,
            loggers=[
                DictLogger(prefix="test/", module=self, metrics=self.test_metrics),
                MetricsTableLogger(
                    prefix="test/",
                    module=self,
                    metrics=self.test_metrics,
                    table_name="Metrics",
                ),
                DataframeLogger(
                    prefix="test/",
                    metric_name="LatentSpace",
                    module=self,
                    data=self.test_latent_space,
                ),
            ],
        )

    def _final_step(self, y_hat):
        """
        Final step of the forward pass. Includes the final activation function if any and the final prediction.
        This should be overriden by the subclass if needed (for example for classification tasks).

        Args:
            y_hat: Predicted outputs.

        Returns:
            Predicted outputs.
        """
        return y_hat

    def training_step(self, batch, batch_idx):
        """
           Training step.

        Args:
            batch: Input batch.
            batch_idx: Index of the current batch.

        Returns:
            Dictionary containing the loss value, ground truth labels, and predicted outputs.
        """
        batch = [batch[0], batch[1][0]]
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch: Input batch.
            batch_idx: Index of the current batch.

        Returns:
            Dictionary containing the loss value, ground truth labels, and predicted outputs.
        """
        batch = [batch[0], batch[1][0]]
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        """
        Test step.

        Args:
            batch: Input batch.
            batch_idx: Index of the current batch.

        Returns:
            Dictionary containing the loss value, ground truth labels, and predicted outputs.
        """
        labels = batch[1][1]
        batch = [batch[0], batch[1][0]]
        latent_vectors = self.model.encoder(batch[0])
        latent_vectors = latent_vectors[-1]

        for latent_vector in latent_vectors:
            self.test_latent_vectors.append(latent_vector.cpu().detach())
        self.test_labels.append(labels.cpu().detach().numpy())
        self.test_images.append(batch[0])

        return super().test_step(batch, batch_idx)

    def on_test_epoch_end(self):
        """
        Callback function called at the end of each testing epoch.
        Computes and logs the testing metrics.
        """
        flatten = [
            latent_vector.flatten().detach().numpy()
            for latent_vector in self.test_latent_vectors
        ]
        df = pd.DataFrame(flatten)
        df.columns = df.columns.astype(str)
        df.index = df.index.astype(str)

        flattened_list = list(chain.from_iterable(self.test_labels))
        df["target"] = np.array(flattened_list)

        if self.class_names is not None:
            df["target"] = df["target"].map(lambda x: self.class_names[x])
        cols = df.columns.astype(str).tolist()
        df = df[cols[-1:] + cols[:-1]]

        self.test_latent_space.append(df.copy())
        super().on_test_epoch_end()

        self.test_latent_vectors = []
        self.test_labels = []
        self.test_images = []
        self.test_latent_space = []


class VariationalAutoencoderLightningModule(BaseLightningModule):
    """
    LightningModule Variational Autoencoder Module.

    Args:
        model: Model instance representing the Network Model.
        model_name: Model's name.
        loss_fn: Loss function used for training.
        metrics: Metrics used for evaluation.
        lr (float): Learning rate for the optimizer.
        scheduler_max_it (int): Maximum number of iterations for the learning rate scheduler.
        weight_decay (float): Weight decay for the optimizer.

    Attributes:
        model: Model instance representing the Convolution Network Model.
        model_name: Model's name.
        loss_fn: Loss function used for training.
        train_metrics: Metrics used for training evaluation.
        val_metrics: Metrics used for validation evaluation.
        test_metrics: Metrics used for testing evaluation.
        lr (float): Learning rate for the optimizer.
        scheduler_max_it (int): Maximum number of iterations for the learning rate scheduler.
    """

    def __init__(
        self,
        model,
        model_name,
        loss_fn,
        metrics,
        lr,
        scheduler_max_it,
        weight_decay=0,
        class_names=None,
    ):
        super(VariationalAutoencoderLightningModule, self).__init__(
            model=model,
            model_name=model_name,
            loss_fn=loss_fn,
            metrics=metrics,
            lr=lr,
            scheduler_max_it=scheduler_max_it,
            weight_decay=weight_decay,
        )
        self.class_names = class_names
        self.test_latent_vectors = []
        self.test_labels = []
        self.test_images = []
        self.test_latent_space = []

        self.configure_metrics_managers()
        self.configure_loggers()

    def configure_metrics_managers(self):
        """
        Configure metrics managers.
        """
        self.train_metrics_manager = MetricsManager(
            module=self, metrics=self.train_metrics
        )
        self.val_metrics_manager = MetricsManager(module=self, metrics=self.val_metrics)
        self.test_metrics_manager = MetricsManager(
            module=self, metrics=self.test_metrics
        )

    def configure_loggers(self):
        """
        Configure loggers.
        """
        self.train_loss_logger = ScalarLogger(
            prefix="train/", module=self, scalar_name="loss"
        )
        self.val_loss_logger = ScalarLogger(
            prefix="val/", module=self, scalar_name="loss"
        )
        self.test_loss_logger = ScalarLogger(
            prefix="test/", module=self, scalar_name="loss"
        )

        self.train_metrics_logger = DictLogger(
            prefix="train/", module=self, metrics=self.train_metrics
        )
        self.val_metrics_logger = DictLogger(
            prefix="val/", module=self, metrics=self.val_metrics
        )
        self.test_metrics_logger = self.test_metrics_logger = LoggerCollection(
            prefix="test/",
            module=self,
            loggers=[
                DictLogger(prefix="test/", module=self, metrics=self.test_metrics),
                MetricsTableLogger(
                    prefix="test/",
                    module=self,
                    metrics=self.test_metrics,
                    table_name="Metrics",
                ),
                DataframeLogger(
                    prefix="test/",
                    metric_name="LatentSpace",
                    module=self,
                    data=self.test_latent_space,
                ),
            ],
        )

    def _common_step(self, batch, batch_idx):
        """
        Common step for training, validation, and testing.

        Args:
            batch: Input batch.
            batch_idx: Index of the current batch.

        Returns:
            Tuple containing the ground truth labels, predicted outputs, and loss value.
        """
        x, y = batch
        y_hat, mean, log = self(x)

        loss = self.loss_fn(y_hat, y, mean, log)
        return y, y_hat, loss
    
    def _final_step(self, y_hat):
        """
        Final step of the forward pass. Includes the final activation function if any and the final prediction.
        This should be overriden by the subclass if needed (for example for classification tasks).

        Args:
            y_hat: Predicted outputs.

        Returns:
            Predicted outputs.
        """
        return y_hat

    def training_step(self, batch, batch_idx):
        """
           Training step.

        Args:
            batch: Input batch.
            batch_idx: Index of the current batch.

        Returns:
            Dictionary containing the loss value, ground truth labels, and predicted outputs.
        """
        batch = [batch[0], batch[1][0]]
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch: Input batch.
            batch_idx: Index of the current batch.

        Returns:
            Dictionary containing the loss value, ground truth labels, and predicted outputs.
        """
        batch = [batch[0], batch[1][0]]
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        """
        Test step.

        Args:
            batch: Input batch.
            batch_idx: Index of the current batch.

        Returns:
            Dictionary containing the loss value, ground truth labels, and predicted outputs.
        """
        labels = batch[1][1]
        batch = [batch[0], batch[1][0]]
        encoded = self.model.encoder(batch[0])
        latent_vectors, _, _ = self.model.bottleneck(encoded)

        for latent_vector in latent_vectors:
            self.test_latent_vectors.append(latent_vector.cpu().detach())
        self.test_labels.append(labels.cpu().detach().numpy())
        self.test_images.append(batch[0])

        return super().test_step(batch, batch_idx)

    def on_test_epoch_end(self):
        """
        Callback function called at the end of each testing epoch.
        Computes and logs the testing metrics.
        """
        flatten = [
            latent_vector.flatten().detach().numpy()
            for latent_vector in self.test_latent_vectors
        ]
        df = pd.DataFrame(flatten)
        df.columns = df.columns.astype(str)
        df.index = df.index.astype(str)

        flattened_list = list(chain.from_iterable(self.test_labels))
        df["target"] = np.array(flattened_list)

        if self.class_names is not None:
            df["target"] = df["target"].map(lambda x: self.class_names[x])
        cols = df.columns.astype(str).tolist()
        df = df[cols[-1:] + cols[:-1]]

        self.test_latent_space.append(df.copy())
        super().on_test_epoch_end()

        self.test_latent_vectors = []
        self.test_labels = []
        self.test_images = []
        self.test_latent_space = []
