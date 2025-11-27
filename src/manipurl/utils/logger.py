import mlflow
import numpy as np
import torch


mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow_client = mlflow.tracking.MlflowClient()


class Logger:

    def __init__(self, run_name = None):
        self.mlflow_run = mlflow.start_run(run_name=run_name)
        self.current_step = 0
        self.run_id = self.mlflow_run.info.run_id

        self.metrics = {}

    def increment(self):
        for key, values in self.metrics.items():
            mlflow_client.log_metric(
                self.run_id, key, 
                torch.as_tensor(values).mean(dtype=torch.float32).cpu().numpy(), 
                step=self.current_step)  

        self.current_step += 1
        self.metrics = {}

    def log_parameter(self, key, value):
        mlflow_client.log_param(self.run_id, key, value)

    def log_parameters(self, params):
        for key, value in params.items():
            mlflow_client.log_param(self.run_id, key, value)

    def log_metric(self, key, value):
        if key in self.metrics:
            self.metrics[key].append(value)
        else:
            self.metrics[key] = [value]
        
    def log_metrics(self, metrics):
        for key, value in metrics.items():
            self.log_metric(key, value)

    def stop(self):
        mlflow.end_run()


class NoLogger:
    def log_parameter(self, key, value):
        pass
    def log_parameters(self, params):
        pass
    def log_metric(self, key, value):
        pass
    def log_metrics(self, metrics):
        pass