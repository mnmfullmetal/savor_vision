import cv2
import time
from ultralytics import YOLO
from utils import get_model_path   

# get model path
MODEL_PATH = get_model_path(model="audit_model", version="audit_v1")

def main():
    model = YOLO(MODEL_PATH) # load the model
    cam = cv2.VideoCapture(0) # open camera feed
    
    if not cam.isOpened():
        print("ERROR: Could not open camera")
        exit()
    
    print(f"MODEL: {MODEL_PATH}")
    print("CAMERA ACTIVE: Press 'Q' to exit.")

    while cam.isOpened():
        start_time = time.time() # set start time to later calculate inference time
        success, frame = cam.read() # read 1 frame from the camera
        if not success:
            print("ERROR: Could not read from camera")
            break
        
        # perform prediction on single frame, recieve results object 
        results = model.predict(source=frame, conf=0.75, iou=0.45, verbose=False)
        
        # calculate time (in ms) it took to receive results
        inference_time = (time.time() - start_time) * 1000

        # check if there was a detection made
        if results[0].boxes:
            count = len(results[0].boxes) # get count of items detected this frame
            print(f"Frame: {inference_time:.1f}ms | Items Found: {count}")
            
            # unpack detection data ( confidence, class index and name)
            for i, box in enumerate(results[0].boxes):
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                name = model.names[cls]
                print(f"  [{i}] {name} - Conf: {conf:.2f}")
        else:
            print(f"Frame: {inference_time:.1f}ms | No Detections")
        
        # draw bounding box 
        annotated_frame = results[0].plot()

        # display live feed window
        cv2.imshow("Savor audit simple - Debug", annotated_frame)
        
        # break while loop if "q" is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    # close camera and live feed window 
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()