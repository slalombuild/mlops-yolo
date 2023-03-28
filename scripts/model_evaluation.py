import yaml
import logging
from scripts.mlflow_utils.register_model import log_metrics

def evaluate_model(save_dir: str, evaluation_thresholds: dict, evaluation_metrics: list):
    """Evaluate the best performing object detection model saved in the specified directory based on pre-defined
    evaluation metrics and threshold values.

     Args:
         save_dir (str): The path to the directory where the results of the detection model are saved.
         evaluation_thresholds (dict): A dictionary of thresholds for metrics that the model must beat.
         evaluation_metrics (list): A list of metrics to compare. Note these metrics need to exist in the model outputs.
     Returns:
         evaluation_results (bool): A bool representing the evaluation results for all evaluation metrics based on the best performing model.
         If the threshold for all metrics passed the value will be True.
    """
    # Load the model's metrics from Yolo's output path
    metrics_list = log_metrics(save_dir, log_results=False)
    # Iterate to find the best Epoch to evaluate on
    best_mAP50 = max(metrics_list, key=lambda x: x["mAP50"])

    # Check that each metric passes threshold
    checks_passed = True
    for param in evaluation_thresholds:
        if best_mAP50[param] < evaluation_thresholds[param]:
            logging.info(
                f"The parameter {param}:{best_mAP50[param]} in the best.pt model does not meet the threshold of {evaluation_thresholds[param]}"
            )
            checks_passed = False
        else:
            logging.info(
                f"The parameter {param}:{best_mAP50[param]} in the best.pt model exceeds the threshold of {evaluation_thresholds[param]}"
            )
    
    return checks_passed
