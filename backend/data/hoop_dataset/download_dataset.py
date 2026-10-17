from roboflow import Roboflow
rf = Roboflow(api_key="WG4OJDyfsF6CM8hT3eKp")
project = rf.workspace("jacob-isaac").project("object-detection-1wdi1")
version = project.version(1)
dataset = version.download("yolov8")
                