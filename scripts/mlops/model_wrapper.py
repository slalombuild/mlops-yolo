"A wrapper around the YOLO model to integrate with MLFlow"
import logging
import numpy as np
import pandas as pd
import mlflow
from ultralytics import YOLO


class YoloWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.results = None
        self.results_df = None
        self.data = None

    def load_context(self, context: object):
        """Load Yolo model from context path

        Args:
            context (object): An MLFlow object that is used to define the path to the model.
        """
        logging.info(f"context.artifacts[path]:{context.artifacts['path']}")
        self.model = YOLO(context.artifacts["path"])

    def reformat_data(self):
        """Reformat given dictionary object that was coerced into numpy arrays into str,int,flow"""
        # For each key-value pair, convert value to string.
        for key, value in self.data.items():
            if isinstance(value, np.ndarray):
                self.data[key] = ",".join(map(str, [value]))

        # For each key-value pair, convert value to appropriate type.
        for key, value in self.data.items():
            if value.isnumeric():  # Check if the value is an integer
                self.data[key] = int(value)
            elif value.replace(".", "", 1).isdigit():  # Check if the value is a float
                self.data[key] = float(value)
            elif value.lower() in ["true", "false"]:  # Check if the value is a boolean
                self.data[key] = value.lower() == "true"

    def yolo_results_to_df(self):
        """Create Yolo results as a df"""
        # Retrieve bounding boxes
        boxes = self.results[0].boxes
        # Map class to string names
        names = []
        for object_class in boxes.cls.numpy():
            names.append(self.model.names[object_class])
        # Create return df
        self.results_df = pd.DataFrame(
            np.c_[boxes.xyxy.numpy(), boxes.conf, boxes.cls.numpy(), np.array(names)],
            columns=["X1", "Y1", "X2", "Y2", "conf", "cls", "names"],
        )

    def predict(self, context: object, data: dict):
        """Wrapper function around Yolo's predict function. Results are returned as a pandas dataframe.

        Args:
            context (object): An MLFlow object that is used to define the path to the model.
            data (dict): dictionary with source for inference and override parameters for the model.

        Returns:
            _type_: _description_
        """
        self.data = data
        logging.info(f"Data input: f{self.data}")
        # Reformat data
        self.reformat_data()
        logging.info(f"Data after reformat: f{self.data}")

        # Pass inputs to predict
        self.results = self.model.predict(**self.data)
        # Transform results to pandas df
        self.yolo_results_to_df()

        return self.results_df
