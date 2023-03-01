import mlflow
import torch
from ultralytics import YOLO
import logging
class TennisDetectorWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        logging.info(f"context.artifacts[path]:{context.artifacts['path']}")
        self.model = YOLO(context.artifacts['path'])

    def predict(self, context, img):
        results = self.model.predict(source=img)

        return results
