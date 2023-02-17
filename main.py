from scripts.utilities import get_roboflow_data
from scripts.argparse import create_parser
import sys
import logging
import json
import os
import glob

# Get arguments:
# Execute the parse_args() method to get arguments
args = create_parser().parse_args()

# Clean log directory
if remove_logs:
    for filename in glob.glob("log/*.log*"):
        os.remove(filename)

# Setting where to write log file
log_path = os.path.join(os.path.dirname(__file__), "log", "data_validation.log")
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
logging.info(f"YOLOv8 Model training log path: {log_path}")

def main():
    logging.info("Starting to download training, validationa and test images")
    get_roboflow_data(data_version = args.data_version, api_key = args.roboflow_api_key)
    logging.info("Image download complete")

if __name__ == "__main__":
    try:
        logging.info("Process Start")
        main()
        logging.info("Process Complete")
    except Exception as e:
        raise Exception(e)