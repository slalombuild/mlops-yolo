
import mlflow
import yaml
from scripts.mlops.register_model import log_metrics

def evaluate_model(save_dir,run_id, config_path):
    # Load the model's metrics from MLflow
    metrics = log_metrics(save_dir)
    
    # Load the thresholds from the YAML config file
    with open(config_path, "r") as config_file:
        thresholds = yaml.safe_load(config_file)
    
    # Check if the metrics are above the thresholds
    if mAP50 < thresholds["mAP50"]:
        return "mAP50 is below threshold"
    elif mAP50_95 < thresholds["mAP50-95"]:
        return "mAP50-95 is below threshold"
    elif precision < thresholds["precision"]:
        return "precision is below threshold"
    elif recall < thresholds["recall"]:
        return "recall is below threshold"
    else:
        return "Model passed evaluation"