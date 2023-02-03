from roboflow import Roboflow
rf = Roboflow(api_key="E1UAwvyKe8uHH4eJGFid")
project = rf.workspace("").project("tennis-tracking")
dataset = project.version(1).download("yolov5")
#curl -L "https://app.roboflow.com/ds/izmMc7WfSm?key=pJ8Li4jZll" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip