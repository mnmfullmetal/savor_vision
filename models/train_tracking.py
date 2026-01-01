from ultralytics import YOLO
import os

def main():
    model = YOLO("yolov8s.pt") 

    data_path = os.path.join(os.getcwd(), "tracking_model", "data.yaml")

    results = model.train(
        data=data_path,
        epochs=100,
        imgsz=640,
        device=0,
        project="tracking_model",
        name="track_v1",
        mosaic=1.0,        
        mixup=0.1,         
        patience=50,       
        close_mosaic=10 
    )

if __name__ == "__main__":
    main()