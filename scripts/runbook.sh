#MAC
yolo task=detect mode=train  model=yolov8m.pt data=data.yaml  epochs=2 imgsz=1920 batch=2
yolo task=detect mode=predict model=best_02_12.pt conf=0.20 source=videos/clips/sample_atp_rallies/sample_atp_rallies_1.mp4 show=true  
yolo task=detect mode=predict model=best_02_16.pt conf=0.20 source=videos/clips/roland_garros/roland_garros_4.mp4 save=true 
python3 track.py --source ../videos/clips/sample_atp_rallies/sample_atp_rallies_7.mp4 --yolo-weights ../best_02_12.pt --tracking-method strongsort --save-vid






# PC
yolo task=detect mode=train  model=yolov8m.pt data=C:\Users\15094\Documents\GitHub\yolo_tennis\data.yaml  epochs=2 imgsz=1920 batch=2
yolo task=detect mode=predict model=best_02_16.pt conf=0.50 source=videos\roland_garros_full_clip.mp4 show=true
python track.py --source ..\videos\roland_garros_full_clip.mp4 --yolo-weights ..\best_02_16.pt --tracking-method strongsort --imgsz 1920  --max-det 3 --conf-thres .35 --show-vid

#Main execution
#Train a model but do not register
python -m main --roboflow_api_key $api_key --get_data False --train_model True --model_evaluation True --register_model False --build_image False   --remove_logs False --logging_level INFO
#Register a  model but do not train.
python -m main --roboflow_api_key $api_key --get_data False --train_model False --model_evaluation True --register_model True --model_path "runs/detect/train" --build_image False --remove_logs False --logging_level INFO



# MLFLOW
 mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 8000 
 mlflow ui
pkill -f gunicorn
python -m scripts.mlops.register_model --name BestTennisDetector --model ultralytics/runs/detect/train/weights/best.pt --model-name TennisDetector
mlflow sagemaker build-and-push-container --no-push 


#MLFLOW build a sagemaker compatible docker container
mlflow models build-docker --name "tennisdetector"
mlflow deployments run-local --target sagemaker \
        --name my-local-deployment \
        --model-uri "mlruns/683127219138585204/dd655572a17c4add9f10f9243a4e5c46/artifacts/model" \
        --flavor python_function \
        -C port=4000 \
        -C image="tennisdetector"
curl -v  -H "Content-Type: application/json" -d @tests/test_data/image.json http://localhost:4000/invocations
docker exec -it container_name bash 
docker cp filepath container_name:filepath
#Key
E1UAwvyKe8uHH4eJGFid
# Running in a container
docker build -t yolov8_mlops .
# Remove --gpus all if no GPUs are available and ipc flag if not running on a EC2 device.
docker run --name yolo_container --gpus all --ipc=host yolov8_mlops --roboflow_api_key $api_key --get_data True --train_model True --model_evaluation True --register_model True --build_image False   --remove_logs False --logging_level INFO
# Prune old images
docker image prune
# Prune old no running containers
docker container prune
# Run the docker image but not the command
docker run -it <image_name>