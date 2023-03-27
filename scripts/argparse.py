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
        "--get_data",
        metavar="get_data",
        type=str2bool,
        required=True,
        choices={True, False},
        help="If true new data will be downloaded.",
    )

    model_trainer_parse.add_argument(
        "--roboflow_api_key",
        metavar="roboflow_api_key",
        type=str,
        required=False,
        help="API key to access Roboflow",
    )

    model_trainer_parse.add_argument(
        "--train_model",
        metavar="train_model",
        type=str2bool,
        required=True,
        choices={True, False},
        help="If true a new model will be trained",
    )

    model_trainer_parse.add_argument(
        "--model_evaluation",
        metavar="model_evaluation",
        type=str2bool,
        required=True,
        choices={True, False},
        help="If true the model will be evaluated",
    )

    model_trainer_parse.add_argument(
        "--register_model",
        metavar="register_model",
        type=str2bool,
        required=True,
        choices={True, False},
        help="If true a a model will be registered",
    )

    model_trainer_parse.add_argument(
        "--model_path",
        metavar="model_path",
        type=str,
        required=False,
        help="A equvalent path of the outputs of the YOLO library in the 'run/train' directory. This will only be applied if training is set to false",
    )

    model_trainer_parse.add_argument(
        "--build_image",
        metavar="build_image",
        type=str2bool,
        required=True,
        choices={True, False},
        help="If true a a docker image will be built from the model",
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
