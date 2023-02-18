
from roboflow import Roboflow
import os
import yaml


def get_roboflow_data(api_key:str,data_version:int ,workspace:str,project:str,data_format:str,write_dataset:str,overwrite:bool) -> str: 
    """Get dataset from Roboflow.

    Args:
        api_key (str, optional): API key to Roboflow environment.
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
    location = os.path.join(write_dataset,project+"-"+str(data_version))
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    dataset = project.version(data_version).download(data_format,location =location, overwrite=overwrite)
    return location

def main():
    """Performs a test-run for local testing"""
    get_roboflow_data(2)


if __name__ == "__main__":
    main()
