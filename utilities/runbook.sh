python3 train.py --img 640 --batch 10 --epochs 15 --data ../data.yaml  --weights yolov5s.pt --name yolov5s_results  --cache
python3 train.py --img 640 --batch 16 --epochs 3 --data ../data.yaml --weights yolov5s.pt
python3 detect.py --weights runs/train/exp/weights/best.pt --source ../laver_cup_2017_edit.mp4 --view-img
ython3 detect.py  --source ../laver_cup_2017_edit.mp4 --view-img               