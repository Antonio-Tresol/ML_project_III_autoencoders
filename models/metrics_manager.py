from abc import ABC, abstractmethod

class BaseMetricsManager():
    def __init__(self, module):
        """
        Initialize the base metrics manager
        
        Args:
            module: The Lightning module that the metrics manager belongs to.
        """
        self.module = module
    
    @abstractmethod   
    def update(self, y_true, y_pred):
        """
        Update the metrics with the new predictions.
        
        Args:
            y_true: The true values.
            y_pred: The predicted values.
        """
        pass
    
    @abstractmethod 
    def reset(self):
        """
        Reset the metrics.
        """
        pass

class MetricsManager(BaseMetricsManager):
    def __init__(self, module, metrics):
        """
        Initialize the metrics manager
        
        Args:
            module: The Lightning module that the metrics manager belongs to.
            metrics: A dictionary of metrics to track.
        """
        super(MetricsManager, self).__init__(module)
        self.metrics = metrics
        
    def update_metrics(self, y_true, y_pred):
        """
        Update the metrics with the new predictions.
        
        Args:
            y_true: The true values.
            y_pred: The predicted values.
        """
        self.metrics.update(y_true, y_pred)
        
    def reset_metrics(self):
        """
        Reset the metrics.
        """
        self.metrics.reset()

class MetricsManagerCollection(BaseMetricsManager):
    def __init__(self, module, managers):
        """
        Initialize the metrics manager collection
        
        Args:
            module: The Lightning module that the metrics manager collection belongs to.
            metrics: A dictionary of metrics to track.
        """
        super(MetricsManagerCollection, self).__init__(module)
        self.metric_managers = managers
            
    def update_metrics(self, y_true, y_pred):
        """
        Update the metrics with the new predictions.
        
        Args:
            y_true: The true values.
            y_pred: The predicted values.
        """
        for manager in self.metric_managers:
            manager.update_metrics(y_true, y_pred)
            
        
    def reset_metrics(self):
        """
        Reset the metrics.
        """
        for manager in self.metric_managers:
            manager.reset_metrics()