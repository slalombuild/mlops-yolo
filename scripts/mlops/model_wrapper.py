import mlflow
import torch


class TennisDetectorWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = torch.hub.load(
            "ultralytics/yolov8", "custom", path=context.artifacts["path"]
        )

    def predict(self, context, img):
        objs = self.model(img).xywh[0]

        return objs.numpy()
