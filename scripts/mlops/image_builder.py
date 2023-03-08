from mlflow.deployments import get_deploy_client
import mlflow
import logging

client = mlflow.MlflowClient()
name = "TennisDetector"

# Get the latest version for the model
version = client.get_latest_versions(name=name)[0].version

# Construct the model URI
model_uri = f"models:/{name}/{version}"
logging.info(f"Model URI: {model_uri}")
print(model_uri)
mlflow.models.build_docker(name="tennisdetector")

client = get_deploy_client("sagemaker")
mlflow.sagemaker.run_local(
    name="local_tennisdetector",
    model_uri=model_uri,
    flavor="python_function",
    config={
        "port": 4000,
        "image": "tennisdetector",
    },
)
