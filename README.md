## MLOps with Yolov8
The goal of this repository is to offer an MLOps solution for the Yolov8 model in AWS. This repository allows you to ingest image data with labels to train a model, evaluate that model, and register that model to MLflow

## Overview of the code

- This python script orchestrates a series of function calls to get data, train a model, evaluate a model, register a model, and build a docker image off the model.

1.  The script starts by parsing the arguments passed to it using the create_parser() function in argparse.py and executing the parse_args() method. The arguments determine the execution flow of the script.

2. The script checks the validity of the arguments passed, raising a ValueError if the --model_path argument is specified when the train_model argument is set to True, but the model is to be registered or evaluated, or if the train_model argument is set to False, but the --model_path argument is not specified when the model is to be registered or evaluated.

3. If the --remove_logs argument is passed, the script deletes all files with the .log extension in the log directory.

4. The script sets up the logging configuration by specifying where to write the log file and the logging format.

5. The main() function starts by reading the default parameters from the model_config.yaml file using the yaml library. The get_data params, training_params, and mlflow_params are read from the file and printed to the log.

6. If the --get_data argument is passed, the script downloads the training, validation, and test images using the get_roboflow_data() function from roboflow_data.py script. The Roboflow API key is required for this operation.

7. If the --train_model argument is passed, the script trains a YOLOv8 model using the train_model() function from train.py script. The dataset_dir and training_params are passed as arguments.

8. If the --model_evaluation argument is passed, the script evaluates the trained or registered model using the evaluate_model() function from the model_evaluation.py script. The evaluation parameters and metrics are read from the model_config.yaml file.

9. If the --register_model argument is passed, the script registers the trained or registered model using the register_model() function from mlflow_utils/register_model.py script. The experiment name, model name, and save directory are read from the model_config.yaml file.

10. If the --build_image argument is passed, the script builds a Docker image using the build_and_run_sagemaker_image() function from mlflow_utils/mlflow_build_image.py script. The model name, image name, version, and Docker port are read from the model_config.yaml file.

11. Finally, the main() function is called, and the script prints Process Start before executing the main() function and Process Complete after the execution.

## Getting Started Using the Repo:
- Install Python >3.9 on your machine.
- Clone the repository to your local machine using git clone: `https://github.com/slalombuild/mlops-yolo`.
- Create a new virtual environment for the project. You can do this using venv by running:
    -  `python -m venv <env-name>`
    -  Activate the virtual environment by running `source <env-name>/bin/activate` (on Linux/Mac) or `<env-name>\Scripts\activate` (on Windows).
- Install the required Python packages by `running pip install -r requirements.txt`
- Using the terminal run the following to set you mlflow uri `export MLFLOW_TRACKING_URI=http://example.amazonaws.com:80
- Update values in `config/model_config.yaml` to suit your model parameters.
- Install Docker to be able to build a Docker image later on.
- (Optional) Ensure AWS roles are configured to Access any external data
Once you have completed the above steps, you can proceed to running the script.
- (Optional) Configure an MLflow server using [MLflow Server CDK](https://github.com/aws-samples/amazon-sagemaker-mlflow-fargate/blob/main/README.md)

## Data
- The training data can be created in any way you want. The import piece of the puzzle is to specify it in Yolov8 format.
- The YOLOv5 model expects the dataset to be in the format of darknet .txt files where each line in the .txt file represents one object instance in the image using the following format: class x_center y_center width height. The folder structure for the train, test, and validation data should follow this format:

```    data/
    └─── train/
    |       ├─── images/
    |       └─── labels/
    |
    └─── test/
    |       ├─── images/
    |       └─── labels/
    |
    └─── val/
            ├─── images/
            └─── labels/
```

- If you are using Roboflow, once annotations are completed, the data is export out of Roboflow in Yolov8 format. Exports create `test`, `train`, and `valid` folders as well as a `data.yaml` that can be read in for training.

### S3 Integration
- TBD
### Output
Video after model inference:
- ![Object detection](/photos/tennis_object_detection.gif)




