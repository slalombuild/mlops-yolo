from mlflow.models import build_docker
from mlflow.deployments import get_deploy_client
import mlflow
import pdb
# build_docker(name="mlflow-pyfunc")

# client = get_deploy_client("sagemaker")
# client.run_local(
#     name="my-local-deployment",
#     model_uri="./mlruns/0/9cc65c3a79034f3aa3f238a67ed426a1/artifacts",
#     flavor="python_function",
#     config={
#         "port": 5000,
#         "image": "mlflow-pyfunc",
#     }
# )


# create a client to access the MLflow tracking server
client = mlflow.MlflowClient()

# loop through all registered models
# NOTE: `filter_string` should be optional, but leaving it as `None` failed to work. 
# Instead, using `"name LIKE '%'"` will match all model names
for model in client.search_registered_models(filter_string="name LIKE '%'"):
    # loop through the latest versions for each stage of a registered model
    for model_version in model.latest_versions:
        print(f"name={model_version.name}; run_id={model_version.run_id}; version={model_version.version}, stage={model_version.current_stage}")


name = 'TennisDetector'

# Get the latest version for the model
version = client.get_latest_versions(name=name)[0].version

# Construct the model URI
model_uri = f'models:/{name}/{version}'

# Load the model
model = mlflow.pyfunc.load_model(model_uri)
#pdb.set_trace()
results = model.predict("photos/no-label.jpg")
print(results)
