import argparse


def create_parser():
    """Creates the argument parser for YOLOv8 Model trainer.

    Returns:
        object: argument parser
    """
    # Create the parser
    model_trainer_parse = argparse.ArgumentParser(
        description="Builds a YOLOv8 model based on training data from Roboflow"
    )

    # Add the arguments
    model_trainer_parse.add_argument(
        "--model",
        metavar="pre_trained_model",
        type=str,
        required=True,
        choices={"yolov8n","yolov8s","yolov8m","yolov8l","yolov8x"},
        help="Trained model that you choose to start training the model with",
    )

    model_trainer_parse.add_argument(
        "--roboflow_api_key",
        metavar="roboflow_api_key",
        type=str,
        required=True,
        help="API key to access Roboflow",
    )
    
    model_trainer_parse.add_argument(
        "--data_version",
        metavar="data_version",
        type=int,
        required=True,
        help="Version of the dataset to grab from Roboflow project",
    )
    
    model_trainer_parse.add_argument(
        "--remove_logs",
        metavar="remove_logs",
        type=str2bool,
        required=False,
        choices={True, False},
        default=False,
        help="Remove all logs from the logs directory",
    )
    
    model_trainer_parse.add_argument(
        "--logging_level",
        metavar="logging_level",
        type=str,
        required=False,
        choices={"INFO", "DEBUG"},
        default="INFO",
        help="Logging level to be used when the python module is executed. Values include: INFO and DEBUG",
    )

    
    return model_trainer_parse


def str2bool(v: str):
    """Converts strings to bools

    Args:
        v (str): Strings that could be booleans

    Raises:
        argparse.ArgumentTypeError:

    Returns:
        bool: True or False depending on input
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
