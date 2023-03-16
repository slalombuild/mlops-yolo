## Image detection for tennis players and balls
The goal of this analysis is to offer a image detection solution to track tennis players and tennis balls. We want to be able to track these objects so we can create metrics for total distance run,ball speed, and court position.

### Gathering Images
- The training data is being generated by downloading Youtube videos. These videos are then edited using Adobe Premiere Pro to subset each video based on when the camera angle changes. This generates a series of clips. Clips with camera angles that show rallies from a vantage point are kept, others are discarded. Once the clips are generated we rename the files and convert the MP4s to JPGs. These JPGs are then uploaded to Roboflow and annotated. These annotations include classes: `player` and `ball`.
- Once annotations are completed, the data is export out of Roboflow in Yolov8 format. Exports create `test`, `train`, and `valid` folders as well as a `data.yaml` that can be read in for training.
### Configuration
Under the configuration folder you will find a `model_config.ymal`. This file can be parameterized to change how the training data is retrieved from Roboflow. In addition model training parameters can be adjusted here as well.
### Get data from Roboflow
To retrieve the data from Roboflow a Roboflow API key will be needed. Once this is attained if you only want to get the data run the command below:
- ```python -m main  --get_data True --roboflow_api_key <key here> --train_model False --register_model False  --remove_logs False --logging_level INFO``` 
### Model Training
Model training was done both on a CPU and a cuda enabled enviroment to interact with a GPU. Some of the best models can be found under the models folder. Later dated models are usually better models. To run your own training session and nothing else simply run:
- ```python -m main --get_data False --train_model True --register_model False  --remove_logs False --logging_level INFO``` 
### MLFlow: Registry and Tracking:
We use MLFlow to register and track our models. To track you Yolov8 model and register the model, simply call:
- ```python -m main --get_data False --train_model False --register_model True  --model_path <path to mode> --remove_logs False --logging_level INFO``` 
### Output
Video after model inference:
- ![Object detection](/photos/tennis_object_detection.gif)

### Resources
- https://www.youtube.com/watch?v=QCG8QMhga9k&ab_channel=Roboflow 
- https://github.com/roboflow/notebooks/blob/main/notebooks/how-to-track-football-players.ipynb
- https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/?utm_campaign=Onboarding&utm_content=YOLOv5+welcome&utm_medium=email_action&utm_source=email
- https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov5-object-detection-on-custom-data.ipynb#scrollTo=1NcFxRcFdJ_O
