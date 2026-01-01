from ultralytics import YOLO
import os 

def main():
    model = YOLO('yolov11s-seg.pt')
    data_path = os.path.join(os.getcwd(), "amodal_model", "data.yaml")

    results = model.train(
        data=data_path,
        epochs=100,
        imgsz=640,
        device=0,          
        project="amodal_model",
        name="amodal_v1",
        batch=8,           
        mask_ratio=4,       
        overlap_mask=False, 
        mixup=0.0,          
        mosaic=0.5,        
        hsv_v=0.2,         
        fliplr=0.5,    
        exist_ok=True       
    )

    best_model = YOLO(os.path.join(os.getcwd(), "amodal_model", "amodal_v1", "weights", "best.pt"))

    best_model.export(format="onnx", opset=12, nms=False, simplfy=True, dynamic=False)


if __name__ == "main":
    main()

    