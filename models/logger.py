from abc import ABC, abstractmethod
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import numpy as np
import torch


class BaseMetricsLogger:
    def __init__(self, prefix, module):
        """
        Base class for logging metrics

        Args:
            prefix (str): Prefix to add to the metric name
            module (pl.LightningModule): PyTorch Lightning Module
        """
        self.prefix = prefix
        self.module = module

    @abstractmethod
    def log(self):
        """
        Abstract method to log the metrics
        """
        pass


class DataframeLogger(BaseMetricsLogger):
    def __init__(self, prefix, module, metric_name, data):
        """
        Dataframe logger

        Args:
            prefix (str): Prefix to add to the metric name
            module (pl.LightningModule): PyTorch Lightning Module
            metric_name (str): Name of the metric
            data: Data to log
        """
        self.prefix = prefix
        self.module = module
        self.metric_name = metric_name
        self.data = data

    def log(self):
        """
        Abstract method to log the metrics
        """
        dataframe = wandb.Table(dataframe=self.data[0])
        self.module.logger.experiment.log({f"{self.prefix}{self.metric_name}": dataframe})

class DictLogger(BaseMetricsLogger):
    def __init__(self, prefix, module, metrics):
        """
        Logger for logging metrics as a dictionary

        Args:
            prefix (str): Prefix to add to the metric name
            module (pl.LightningModule): PyTorch Lightning Module
            metrics: Metrics object to compute the metrics
        """
        super(DictLogger, self).__init__(prefix, module)
        self.metrics = metrics

    def log(self):
        """
        Log the metrics as a dictionary
        """
        self.module.log_dict(self.metrics.compute(), on_step=False, on_epoch=True)


class ScalarLogger(BaseMetricsLogger):
    def __init__(self, prefix, module, scalar_name):
        """
        Logger for logging metrics as a scalar

        Args:
            prefix (str): Prefix to add to the metric name
            module (pl.LightningModule): PyTorch Lightning Module
            scalar_name (str): Name of the scalar
        """
        super(ScalarLogger, self).__init__(prefix, module)
        self.scalar_name = scalar_name

    def log(self, metric):
        """
        Log the metric as a scalar

        Args:
            metric: Metric to log
        """
        self.module.log(
            f"{self.prefix}{self.scalar_name}",
            metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )


class ConfusionMatrixLogger(BaseMetricsLogger):
    def __init__(self, prefix, module, y_true_ref, y_pred_ref):
        """
        Logger for logging confusion matrix

        Args:
            prefix (str): Prefix to add to the metric name
            module (pl.LightningModule): PyTorch Lightning Module
            y_true_ref (torch.Tensor): Reference to the true labels
            pred_ref (torch.Tensor): Reference to the predicted labels
        """
        super(ConfusionMatrixLogger, self).__init__(prefix, module)
        self.y_true = y_true_ref
        self.pred = y_pred_ref

    def log_cm(self, y_true, pred):
        """
        Log the confusion matrix

        Args:
            y_true (np.array): True labels
            pred (np.array): Predicted labels
        """
        figsize = (40, 30)
        plt.rcParams["font.size"] = 12
        cm = confusion_matrix(y_true, pred)
        plot = plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt="g",
            xticklabels=self.module.class_names,
            yticklabels=self.module.class_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{self.module.model_name} Confusion Matrix")

        # Log the image to wandb
        self.module.logger.experiment.log(
            {f"{self.prefix}ConfusionMatrix": [wandb.Image(plot)]}
        )

    def log(self):
        """
        Log the confusion matrix
        """
        self.log_cm(
            torch.cat(self.y_true).cpu().detach().numpy(),
            torch.cat(self.pred).cpu().detach().numpy(),
        )


class MetricsTableLogger(BaseMetricsLogger):
    def __init__(self, prefix, module, metrics, table_name):
        """
        Logger for logging metrics as a table

        Args:
            prefix (str): Prefix to add to the metric name
            module (pl.LightningModule): PyTorch Lightning Module
            metrics: Metrics object to compute the metrics
            table_name (str): Name of the table
        """
        super(MetricsTableLogger, self).__init__(prefix, module)
        self.metrics = metrics
        self.table_name = table_name

    def log_table(self, data, columns):
        """
        Log the table

        Args:
            data (list): List of data
            columns (list): List of column names
        """
        table = wandb.Table(data=data, columns=columns)
        self.module.logger.experiment.log({f"{self.prefix}{self.table_name}": table})

    def log(self):
        """
        Log the metrics as a table
        """
        metrics = self.metrics.compute()
        self.log_table(data=[list(metrics.values())], columns=list(metrics.keys()))


class PerClassLogger(BaseMetricsLogger):
    def __init__(self, prefix, module, metrics):
        """
        Logger for logging metrics per class

        Args:
            prefix (str): Prefix to add to the metric name
            module (pl.LightningModule): PyTorch Lightning Module
            metrics: Metrics object to compute the metrics
        """
        super(PerClassLogger, self).__init__(prefix, module)
        self.table_name = "PerClassMetrics"
        self.metrics = metrics

    def log_table(self, data, columns):
        """
        Log the table

        Args:
            data (list): List of data
            columns (list): List of column names
        """
        table = wandb.Table(data=data, columns=columns)
        self.module.logger.experiment.log({f"{self.prefix}{self.table_name}": table})

    def log(self):
        """
        Log the metrics per class
        """
        metrics = self.metrics.compute()

        values = []
        metrics_names = []
        for metric, value in metrics.items():
            metrics_names.append(metric)
            values.append(value.cpu().numpy())

        values = np.transpose(values)
        index_names = self.module.class_names
        data = [
            [index] + [str(val) for val in row]
            for index, row in zip(index_names, values)
        ]
        columns = ["Class"] + metrics_names

        self.log_table(data=data, columns=columns)


class LoggerCollection(BaseMetricsLogger):
    def __init__(self, prefix, module, loggers):
        """
        Collection of loggers

        Args:
            prefix (str): Prefix to add to the metric name
            module (pl.LightningModule): PyTorch Lightning Module
            loggers (list): List of loggers
        """
        super().__init__(prefix, module)
        self.loggers = loggers

    def log(self):
        """
        Log the metrics
        """
        for logger in self.loggers:
            logger.log()
