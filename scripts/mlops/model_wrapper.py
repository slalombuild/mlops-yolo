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
        self.model = YOLO(context.artifacts['path'])

    def predict(self, context, data):
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = ",".join(map(str, [value]))
        with open("tennis_logger_input.txt", "a") as f:
            print(data,file=f)

        results = self.model.predict(**data)
        boxes = results[0].boxes
        names = []
        for x in boxes.cls.numpy():
            names.append(self.model.names[x])
        df_results = pd.DataFrame(np.c_[boxes.xyxy.numpy(),boxes.conf,boxes.cls.numpy(),np.array(names)],columns = ["X1","Y1","X2","Y2","conf","cls","names"])
        json_results = df_results.to_json()

        with open("tennis_logger_results.txt", "a") as f:
            print(json_results,file=f)

        return df_results

