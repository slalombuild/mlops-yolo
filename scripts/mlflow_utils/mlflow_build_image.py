from mlflow.deployments import get_deploy_client
import mlflow
import logging

def build_and_run_sagemaker_image(model_name:str,image_name:str,version:int, docker_port:int):
    """Creates a sagemaker compatible docker image and then runs it.

    Args:
        model_name (str): Name of the MLflow model
        image_name (str): Name to be given to the docker image
        version (int): Version of the model from which to build the docker image
        docker_port (int): Port to deploy the model to.
    """

    client = mlflow.MlflowClient()

    # Get the latest version for the model
    version = client.get_latest_versions(name=model_name)[0].version
    # Construct the model URI
    model_uri = f"models:/{model_name}/{version}"
    logging.info(f"Model URI: {model_uri}")
    mlflow.models.build_docker(name=image_name)

    client = get_deploy_client("sagemaker")
    mlflow.sagemaker.run_local(
        name=image_name+"_local",
        model_uri=model_uri,
        flavor="python_function",
        config={
            "port": docker_port,
            "image": image_name,
        },
    )

if __name__ == "__main__":
    logging.info("Process Start")
    build_and_run_sagemaker_image(model_name = "TennisDetector",image_name ="tennisdetector",version = 0, docker_port = 4000)
    logging.info("Process Complete")
