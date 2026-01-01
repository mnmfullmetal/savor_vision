import cv2
import time
from ultralytics import YOLO
from utils import get_model_path

# get model path
MODEL_PATH = get_model_path(model="tracking_model", version="track_v1")

def main():
    
    model = YOLO(MODEL_PATH) # load the model
    cam = cv2.VideoCapture(0) # open camera feed

    if not cam.isOpened():
        print("ERROR: Could not open camera")
        exit()
        
    print(f"Model: {MODEL_PATH}")
    print("CAMERA STARTED: Press 'Q' to exit.")
    
    while cam.isOpened():
        start_time = time.time() # set start time to later calculate inference time
        success, frame = cam.read()  # read 1 frame from the camera
        if not success:
            print("ERROR: Could not read from camera")
            break

        # perform tracking on single frame, recieve results object    
        results = model.track(frame, persist=True, tracker="savor_tracker.yaml", conf=0.35, verbose=False)
        
        # calculate time (in ms) it took to receive results
        inference_time = (time.time() - start_time) * 1000 
        
        # check if there was a detection made
        if results[0].boxes and results[0].boxes.id is not None:

            # send pytorch tensors to cpu and format for std python math
            ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()
            
            print(f"Frame Processed ({inference_time:.1f}ms) | Objects Detected: {len(ids)}")
            
            # create zipper object and unpack tuples (detection info)
            for id, conf, cls in zip(ids, confs, classes):
                name = model.names[cls] # get class name using class index
                print(f"  [ID {id}] {name} | Conf: {conf:.2f}")
        else:
            print(f"Frame Processed ({inference_time:.1f}ms) | *** NO OBJECTS TRACKED ***")

        # draw bounding boxes with info text
        annotated_frame = results[0].plot()
        cv2.putText(annotated_frame, f"Latency: {inference_time:.1f}ms", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # display live feed window 
        cv2.imshow("Savor tracking simple - Debug", annotated_frame)
        
        # break while loop if "q" is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # close camera and live feed window    
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()