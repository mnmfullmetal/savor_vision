from ultralytics import YOLO
from tests.utils import find_data_path
import os

def main():
    
    model = YOLO("yolov8n.pt")

    data_path = find_data_path()

    results = model.train(
        data=data_path,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        project="Savor_training",
        name="mvp_run_2",
    )

    model.export(format="onnx", opset=11)

if __name__ == "__main__":
    main()