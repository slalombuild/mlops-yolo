#MAC
yolo task=detect mode=train  model=yolov8m.pt data=data.yaml  epochs=2 imgsz=1920 batch=2
yolo task=detect mode=predict model=best_02_12.pt conf=0.20 source=videos/clips/sample_atp_rallies/sample_atp_rallies_1.mp4 show=true  
python3 track.py --source ../videos/clips/sample_atp_rallies/sample_atp_rallies_7.mp4 --yolo-weights ../best_02_12.pt --tracking-method strongsort --save-vid




# PC
yolo task=detect mode=train  model=yolov8m.pt data=C:\Users\15094\Documents\GitHub\yolo_tennis\data.yaml  epochs=2 imgsz=1920 batch=2
yolo task=detect mode=predict model=best_02_16.pt conf=0.50 source=videos\roland_garros_full_clip.mp4 show=true
python track.py --source ..\videos\roland_garros_full_clip.mp4 --yolo-weights ..\best_02_16.pt --tracking-method strongsort --imgsz 1920  --max-det 3 --conf-thres .35 --show-vid