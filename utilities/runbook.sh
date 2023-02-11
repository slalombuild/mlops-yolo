python3 train.py --img 640 --batch 10 --epochs 15 --data ../data.yaml  --weights yolov5s.pt --name yolov5s_results  --cache
python3 train.py --img 640 --batch 16 --epochs 3 --data ../data.yaml --weights yolov5s.pt
python3 detect.py --weights runs/train/exp/weights/best.pt --source ../laver_cup_2017_edit.mp4 --view-img
python3 detect.py  --source ../laver_cup_2017_edit.mp4 --view-img    
  yolo task=detect mode=predict model=best.pt conf=0.50 source=videos/paris_masters_full_clip.mp4  show=true    