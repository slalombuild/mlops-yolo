import argparse
import cloudpickle
import mlflow
import os
from pathlib import Path
import torch
import csv
import yaml
import logging
from sys import version_info
from scripts.mlops import model_wrapper
from scripts.mlops.model_wrapper import TennisDetectorWrapper

mlflow.set_tracking_uri("./mlruns")


def get_experiment_id(name):
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        return exp_id
    return exp.experiment_id


def read_lines(path):
    with open(path) as f:
        return f.read().splitlines()


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


def get_path_w_extension(path, extension, ignore_files=[]):
    logging.debug(f"Path: {path}")
    logging.debug(f"Extension: {extension}")
    if isinstance(path, str):
        abs_path = os.path.abspath(path)
    elif isinstance(path, Path):
        abs_path = path.absolute()
    else:
        return f"Error: Path {path} is not valid."

    if not os.path.exists(abs_path):
        return f"Error: Path {abs_path} does not exist."

    if os.path.isdir(abs_path):
        pt_files = []
        for root, dirs, files in os.walk(abs_path):
            for file in files:
                if (
                    file.endswith(extension)
                    and os.path.basename(file) not in ignore_files
                ):
                    pt_files.append(os.path.join(root, file))
        if len(pt_files) == 1:
            return pt_files[0]
        elif len(pt_files) > 1:
            return f"Error: Multiple {extension} files found in directory {abs_path}. Please specify a more specific path."
        else:
            return f"Error: No {extension} files found in directory {abs_path}."
    elif os.path.isfile(abs_path) and abs_path.endswith(extension):
        return abs_path
    else:
        return f"Error: Path {abs_path} is not a valid directory or {extension} file."


def register_model(experiment_name: str, model_name: str, save_dir: Path):
    """Registers a model with mlflow

    Args:
        experiment_name (str): Name of Mlfow experiment
        model_name (str): Name that will be registered with Mlflow
        save_dir (Path): Path object where the results of the Yolo model are saved. I.e 'runs' directory
    """
    save_dir = Path(save_dir)
    logging.debug(f"Save Directory: {save_dir}")

    model_path = get_path_w_extension(
        path=save_dir, extension=".pt", ignore_files=["last.pt"]
    )
    artifacts = {"path": model_path}

    model = TennisDetectorWrapper()

    exp_id = get_experiment_id(experiment_name)

    cloudpickle.register_pickle_by_value(model_wrapper)

    with mlflow.start_run(experiment_id=exp_id) as run:
        # Log some params
        with open(save_dir / "args.yaml", "r") as param_file:
            params = yaml.safe_load(param_file)
        mlflow.log_params(params)
        log_metrics(save_dir)
        for file in sorted(save_dir.glob("*.png")):
            mlflow.log_artifact(file)
        pip_reqs = read_lines("requirements.txt")
        mlflow.pyfunc.log_model(
            "model",
            python_model=model,
            pip_requirements=pip_reqs,
            artifacts=artifacts,
            registered_model_name=model_name,
        )
        run_id = run.info.run_uuid
        experiment_id = run.info.experiment_id
        mlflow.end_run()
        logging.info(f"artifact_uri = {mlflow.get_artifact_uri()}")
        logging.info(f"runID: {run_id}")


def main():
    pass  # TODO


if __name__ == "__main__":
    main()
