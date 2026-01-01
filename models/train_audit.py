from ultralytics import YOLO
import os

def main():
    model = YOLO("yolov8s-seg.pt") 

    data_path = os.path.join(os.getcwd(), "audit_model", "data.yaml")

    results = model.train(
        data=data_path,
        epochs=100,
        imgsz=640,
        device=0,
        project="audit_model",
        name="audit_v1",
        mask_ratio=1,
        overlap_mask=False,
        mixup=0.0,         
        mosaic=0.5,        
        hsv_v=0.2,     
        fliplr=0.5        
    )

    model.export(format="onnx", opset=12)

if __name__ == "__main__":
    main()