from scripts.roboflow_data import get_roboflow_data
from scripts.argparse import create_parser
from scripts.train import train_model
from scripts.mlops.register_model import register_model
from scripts.mlops.mlflow_build_image import build_and_run_sagemaker_image
import sys
import logging
import json
import os
import glob
import yaml

# Get arguments:
# Execute the parse_args() method to get arguments
args = create_parser().parse_args()

if args.train_model == True and args.register_model == True and args.model_path != None:
    raise ValueError("Argument: --model_path can only be specified if the argument 'train_model' is set to False when a model will be registered")
if args.train_model == False and args.register_model == True and args.model_path == None:
    raise ValueError("If the argument: 'train_model' is set to False then the optional argument --model_path must be specified when a model will be registered")

# Clean log directory
if args.remove_logs:
    for filename in glob.glob("log/*.log*"):
        try:
            os.remove(filename)
        except OSError:
            pass

# Setting where to write log file
log_path = os.path.join(os.path.dirname(__file__), "log", "train_model.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)


# Setting the basic configuration of the log file to write to a file and to the console
logging.basicConfig(
    handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging_level = logging.getLevelName(args.logging_level)
logging.getLogger().setLevel(args.logging_level)
logging.debug(f"YOLOv8 Model training log path: {log_path}")


def main():
    # Get Default parameters
    with open("config/model_config.yaml", "r") as file:
        yaml_inputs = yaml.safe_load(file)
    roboflow_params = yaml_inputs["roboflow_params"]
    dataset_dir = os.path.join(
        yaml_inputs["roboflow_params"]["write_dataset"],
        yaml_inputs["roboflow_params"]["project"]
        + "-"
        + str(yaml_inputs["roboflow_params"]["data_version"]),
    )
    logging.debug(
        f"Roboflow parameters: {json.dumps(roboflow_params, indent=4, sort_keys=True)}"
    )
    training_params = yaml_inputs["training_params"]
    logging.debug(
        f"Training parameters: {json.dumps(training_params, indent=4, sort_keys=True)}"
    )
    mlflow_params = yaml_inputs["mlflow_params"]
    logging.debug(
        f"MLflow parameters: {json.dumps(mlflow_params, indent=4, sort_keys=True)}"
    )
    if args.get_data:
        if args.roboflow_api_key:
            logging.info("Starting to download training, validation and test images")
            get_roboflow_data(api_key=args.roboflow_api_key, **roboflow_params)
            logging.info(f"Photo download complete at: {dataset_dir}")
        else:
            raise ValueError("Specify a Roboflow API key if parameter 'get_data' is set to True")
    if args.train_model:
        logging.info(
            "Starting a training job. For details around configurations see 'config/model_config.yaml"
        )
        model = train_model(dataset_dir=dataset_dir, **training_params)
        logging.info("Training complete.")
    if args.register_model:
        logging.info(
            f"Registering the model: {model.trainer.save_dir if args.train_model else args.model_path}"
        )
        # Register the model
        register_model(
        experiment_name=mlflow_params["experiment_name"],
        model_name=mlflow_params["model_name"],
        save_dir=model.trainer.save_dir if args.train_model else args.model_path)
    if args.build_image:
        logging.info(
            f"Building an image for model: {mlflow_params['model_name']} "
        )
        build_and_run_sagemaker_image(mlflow_params['model_name'],mlflow_params['image_name'],mlflow_params['version'],mlflow_params['docker_port'],)


if __name__ == "__main__":
    logging.info("Process Start")
    main()
    logging.info("Process Complete")
