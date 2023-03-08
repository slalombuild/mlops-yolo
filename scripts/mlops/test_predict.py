import mlflow
import pdb

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

data = {"source":"https://raw.githubusercontent.com/lhuddleston16/yolo_tennis/main/photos/court_detection.jpg","imgsz":"1920","save_txt":"True"}
results = model.predict(data = data)
print(results)

