from scripts.utilities import update_yaml
from roboflow import Roboflow
import logging
import os
import yaml
import json


def get_roboflow_data(
    api_key: str,
    repo_name: str,
    data_version: int,
    workspace: str,
    project: str,
    data_format: str,
    write_dataset: str,
    overwrite: bool,
) -> str:
    """Get dataset from Roboflow.

    Args:
        api_key (str, optional): API key to Roboflow environment.
        repo_name (str): Name of your repo. This is used to create the path for the YOLO model.
        data_version (int): Version of the dataset that you need to grab from Roboflow.
        workspace (str, optional): Workspace where the dataset lives. Defaults to "slalom".
        project (str, optional): Project for the dataset. Defaults to "tennis-object-detection".
        data_format (str, optional): The data format for the model. Defaults to "yolov8".
        write_dataset (str): location where to write dataset
        overwrite (bool): If true the dataset will be overwritten if project and version re-downloaded.

    Returns:
        str: location of the dataset
    """
    # Ensuring dataset directory exists
    if not os.path.exists(write_dataset):
        os.makedirs(write_dataset)
    dataset_dir = os.path.join(write_dataset, project + "-" + str(data_version))
    logging.info(f"Dataset Directory: {dataset_dir}")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    dataset = project.version(data_version).download(
        data_format, location=dataset_dir, overwrite=overwrite
    )
    updates_to_yaml = {
        "test": os.path.abspath(os.path.join(dataset_dir,"test/images")),
        "train": os.path.abspath(os.path.join(dataset_dir,"train/images")),
        "val": os.path.abspath(os.path.join(dataset_dir,"valid/images")),
    }
    logging.info(
        f"YAML updates: {json.dumps(updates_to_yaml, indent=4, sort_keys=True)}"
    )
    update_yaml(os.path.join(dataset_dir, "data.yaml"), updates_to_yaml)

    return dataset_dir


def main():
    """Performs a test-run for local testing"""
    get_roboflow_data(2)


if __name__ == "__main__":
    main()
