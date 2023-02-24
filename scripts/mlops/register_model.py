import argparse
import cloudpickle
import mlflow
import os
from pathlib import Path
import torch
import csv
import yaml
from sys import version_info
from scripts.mlops import model_wrapper
from scripts.mlops.model_wrapper import TennisDetectorWrapper

mlflow.set_tracking_uri("./mlruns")

PYTHON_VERSION = "{major}.{minor}.1".format(
    major=version_info.major, minor=version_info.minor
)

env = {
    "channels": ["defaults"],
    "dependencies": [
        "python~={}".format(PYTHON_VERSION),
        "pip",
        {
            "pip": [
                "mlflow==2.1.1",
                "ultralytics==8.0.20",
                "cloudpickle=={}".format(cloudpickle.__version__),
                "roboflow==0.2.31",
            ],
        },
    ],
    "name": "tennis_env",
}


def get_experiment_id(name):
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        return exp_id
    return exp.experiment_id


def log_metrics(save_dir):
    with open(save_dir / "results.csv", "r") as csv_file:
        metrics_reader = csv.DictReader(csv_file)
        for metrics in metrics_reader:
            # Create an empty dictionary to store the updated key-value pairs for this row
            updated_metrics = {}
            # Iterate through the key-value pairs in this row's dictionary
            for key, value in metrics.items():
                # Remove whitespace from the key
                key = key.strip()
                value = value.strip()
                # Remove the pattern '(B)' from the key
                key = key.replace("(B)", "")
                # Add the updated key-value pair to the updated row dictionary
                updated_metrics[key] = float(value)
                mlflow.log_metrics(updated_metrics)


def register_model(experiment_name: str, model_name: str, save_dir: Path):
    """Registers a model with mlflow

    Args:
        experiment_name (str): Name of Mlfow experiment
        model_name (str): Name that will be registered with Mlflow
        save_dir (Path): Path object where the results of the Yolo model are saved. I.e 'runs' directory
    """
    model_path = save_dir / "weights/best.pt"
    artifacts = {"path": model_path.absolute().as_posix()}

    model = TennisDetectorWrapper()

    exp_id = get_experiment_id(experiment_name)

    cloudpickle.register_pickle_by_value(model_wrapper)

    with mlflow.start_run(experiment_id=exp_id):
        # Log some params
        with open(save_dir / "args.yaml", "r") as param_file:
            params = yaml.safe_load(param_file)
        mlflow.log_params(params)
        log_metrics(save_dir)
        for file in sorted(save_dir.glob("*.png")):
            mlflow.log_artifact(file)
        mlflow.pyfunc.log_model(
            "model_env",
            python_model=model,
            conda_env=env,
            artifacts=artifacts,
            registered_model_name=model_name,
        )

def main():
    pass # TODO
if __name__ == "__main__":
    main()
