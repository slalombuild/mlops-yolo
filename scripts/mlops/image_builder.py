from mlflow.models import build_docker
from mlflow.deployments import get_deploy_client
import mlflow
import logging
import pdb

client = mlflow.MlflowClient()
name = 'TennisDetector'

# Get the latest version for the model
version = client.get_latest_versions(name=name)[0].version

# Construct the model URI
model_uri = f'models:/{name}/{version}'
logging.info(f'Model URI: {model_uri}')
print(model_uri)
# build_docker(name="mlflow-pyfunc")

# client = get_deploy_client("sagemaker")
# #pdb.set_trace()
# client.run_local(
#     name="my-local-deployment",
#     model_uri=model_uri,
#     flavor="python_function",
#     config={
#         "port": 5000,
#         "image": "mlflow-pyfunc",
#     }
# )