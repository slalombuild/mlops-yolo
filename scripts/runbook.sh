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
python -m main --roboflow_api_key E1UAwvyKe8uHH4eJGFid --get_data True --train_model True --remove_logs False --logging_level INFO


# MLFLOW
 mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 8000 
 mlflow ui
pkill -f gunicorn
python -m scripts.mlops.register_model --name BestTennisDetector --model ultralytics/runs/detect/train/weights/best.pt --model-name TennisDetector
