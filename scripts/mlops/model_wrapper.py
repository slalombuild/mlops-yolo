import mlflow
import torch
from ultralytics import YOLO

YOLO()

class TennisDetectorWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = YOLO(context.artifacts['path'])

    def predict(self, context, img):
        results = self.model.predict(source=img)

        return results
