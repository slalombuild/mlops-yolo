import mlflow
import torch
from ultralytics import YOLO
import logging
import numpy as np
from numpy import array
import pandas as pd
import pdb


class TennisDetectorWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        logging.info(f"context.artifacts[path]:{context.artifacts['path']}")
        self.model = YOLO(context.artifacts["path"])

    def predict(self, context, data):
        # Log input data
        with open("input_data.txt", "a") as f:
            print(data, file=f)

        # For each key-value pair, convert value to string.
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = ",".join(map(str, [value]))

        # For each key-value pair, convert value to appropriate type.
        for key, value in data.items():
            if value.isnumeric():  # Check if the value is an integer
                data[key] = int(value)
            elif value.replace(".", "", 1).isdigit():  # Check if the value is a float
                data[key] = float(value)
            elif value.lower() in ["true", "false"]:  # Check if the value is a boolean
                data[key] = value.lower() == "true"

        # Log converted data
        with open("converted_data.txt", "a") as f:
            print(data, file=f)

        # Pass inputs to predict
        results = self.model.predict(**data)
        # Retrieve bounding boxes
        boxes = results[0].boxes
        # Map class to string names
        names = []
        for x in boxes.cls.numpy():
            names.append(self.model.names[x])
        # Create return df
        df_results = pd.DataFrame(
            np.c_[boxes.xyxy.numpy(), boxes.conf, boxes.cls.numpy(), np.array(names)],
            columns=["X1", "Y1", "X2", "Y2", "conf", "cls", "names"],
        )

        return df_results
