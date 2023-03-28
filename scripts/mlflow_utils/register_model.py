"Registers model in MLFlow"
import os
from pathlib import Path
import csv
import logging
import yaml
import cloudpickle
import mlflow
from scripts.mlflow_utils import model_wrapper
from scripts.mlflow_utils.model_wrapper import YoloWrapper


def get_experiment_id(name: str):
    """Retrieve experiment if registered name, else create experiment.

    Args:
        name (str): Mlflow experiment name

    Returns:
        str: Mlfow experiment id
    """
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        return exp_id
    return exp.experiment_id


def read_lines(path: str):
    """Given a path to a file, this function reads file, seperates the lines and returns a list of those separated lines.

    Args:
        path (str): Path to file

    Returns:
        list: List made of of file lines
    """
    with open(path) as f:
        return f.read().splitlines()


def log_metrics(save_dir: str, log_results: bool = True):
    """Log metrics to Mlflow from the Yolo model outputs.

    Args:
        save_dir (str): Path to Yolo save directory, i.e - runs/train
        log_results (bool): If True, the results are logged to MLflow server
    """
    save_dir = Path(save_dir)
    try:
        with open(save_dir / "results.csv", "r") as csv_file:
            metrics_reader = csv.DictReader(csv_file)
            metrics_list = []
            for metrics in metrics_reader:
                # Create an empty dictionary to store the updated key-value pairs for this row
                updated_metrics = {}
                # Iterate through the key-value pairs in this row's dictionary
                for key, value in metrics.items():
                    # Remove whitespace from the key
                    key = key.strip()
                    value = value.strip()
                    # Remove extra strings in keys
                    patterns = ["(B)", "metrics/"]
                    for pattern in patterns:
                        key = key.replace(pattern, "")
                    # Add the updated key-value pair to the updated row dictionary
                    try:
                        # Add the updated key-value pair to the updated row dictionary
                        updated_metrics[key] = float(value)
                    except ValueError:
                        logging.error(f"ValueError: Could not convert {value} to float.")
                    metrics_list.append(updated_metrics)
                    if log_results:
                        mlflow.log_metrics(updated_metrics)
        return metrics_list
    except FileNotFoundError:
        print(f"FileNotFoundError: Could not find {save_dir / 'results.csv'}.")
    except IOError:
        print(f"IOError: Could not read {save_dir / 'results.csv'}.")


def get_path_w_extension(
    path: str, extension: str, limit: int, ignore_files: list = []
):
    """Finds files that match extensions and returns a list of those files that match up to thhe limit while ignoring files in the ignore_files list.

    Args:
        path (str): Directory to search for files in.
        extension (str): Type of extension to look for.
        ignore_files (list, optional): Specify list of files that will be ignored. Defaults to [].

    Returns:
        list: List of paths to files with extensions.
    """
    logging.debug(f"Path: {path}")
    logging.debug(f"Extension: {extension}")
    if isinstance(path, str):
        abs_path = os.path.abspath(path)
    elif isinstance(path, Path):
        abs_path = path.absolute()
    else:
        raise ValueError(f"Error: Path {path} is not valid.")

    if not os.path.exists(abs_path):
        raise ValueError(f"Error: Path {abs_path} does not exist.")

    if os.path.isdir(abs_path):
        pt_files = []
        for root, dirs, files in os.walk(abs_path):
            for file in files:
                if (
                    file.endswith(extension)
                    and os.path.basename(file) not in ignore_files
                ):
                    pt_files.append(os.path.join(root, file))
        if len(pt_files) <= limit and len(pt_files) > 0:
            return pt_files
        if len(pt_files) > limit:
            raise ValueError(
                f"Error: Given limit: {limit} while number of files found with {extension} extension in directory {abs_path} is {len(pt_files)}"
            )
        else:
            raise FileNotFoundError(
                f"Error: No {extension} files found in directory {abs_path}."
            )
    elif os.path.isfile(abs_path) and abs_path.endswith(extension):
        return [abs_path]
    else:
        raise ValueError(
            f"Error: Path {abs_path} is not a valid directory or {extension} file."
        )


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
        path=save_dir, extension=".pt", limit=1, ignore_files=["last.pt"]
    )[0]
    artifacts = {"path": model_path}

    model = YoloWrapper()

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
        logging.info(f"experiment_id: {experiment_id}")


def main():
    pass  # TODO


if __name__ == "__main__":
    main()
