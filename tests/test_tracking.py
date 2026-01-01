from ultralytics import YOLO
import numpy 
import cv2
from utils import get_model_path
import math
import time

MODEL_PATH = get_model_path(model="tracking_model", version="track_v1") # get model path
PERSISTENCE_THRESHOLD = 10 # amount of frames an item has to be detected before it is considered 
DROPOUT_THRESHOLD = 60 # amount of frames a tracked item can be "lost" and be considered in limbo (neither there nor gone) 
MOVEMENT_THRESHOLD = 5 # distance in pixels an object can move before being considered "moving" 
REASSOCIATION_RADIUS = 250 # distance in pixels an object can relocate and still be re-identified as the same object

tracking_history = {} # permenant record of confirmed items tracked
id_map = {} # to map AI-generated IDs to persistent savor IDs

def main():
    model = YOLO(MODEL_PATH) # load model 
    cam = cv2.VideoCapture(0) # open camera
    if not cam.isOpened():
        print("ERROR: Could not open camera")
        exit()

    print(f"TRACKING MODEL: {MODEL_PATH}")
    print("CAMERA ACTIVE: Press Q to exit")

    while cam.isOpened():
        start_time = time.time()
        success, frame = cam.read() # read one frame at a time from the camera feed
        if not success:
            break
        
        # perform tracking on single frame and recieve results object
        results = model.track(source=frame, conf=0.35, iou=0.5, tracker="savor_tracker.yaml", persist=True, verbose=False)
        inference_time = (time.time() - start_time) * 1000 # calculate the time it takes to detect object
        potential_matches = list(tracking_history.keys()) # snapshot of known IDs (tracking history) to use for spatial re-association
        reclaimed_this_frame = set() # list for IDs re-detected and "reclaimed" on this frame
        current_frame_detections = [] # list for detection IDs seen on this frame

        # check if there has been a detection
        if results[0].boxes and results[0].boxes.id is not None:

            # move pytorch tensors to cpu then convert to formats compatible with std python math
            boxes = results[0].boxes.xywh.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().tolist() 
            classes = results[0].boxes.cls.int().cpu().tolist()
            scores = results[0].boxes.conf.cpu().numpy()
            
            print(f"Frame Processed ({inference_time:.1f}ms) | Detections: {len(ids)}")

            # create zipper object and unpack tuples (detection data) for each detection
            for box, ai_id, class_index, conf in zip(boxes, ids, classes, scores):
                new_x, new_y, new_w, new_h = box # get current bounding box cordinates
                class_name = model.names[class_index] # get class name using class index 

                # check if the id given by the ai is an alias for an existing app level (savor) id 
                if ai_id in id_map:
                    tracking_id = id_map[ai_id] # retrieve the persistent savor id
                    if tracking_id != ai_id: # if ids are different a correction has taken place, print debug
                        print(f"  [MAP] AI ID {ai_id} -> Savor ID {tracking_id}")
                else:
                    tracking_id = ai_id # no alias found, treat as new id for now

                ## SPATIAL RE-ASSOCIATION ##
                # if current detection is new, check if its a returning id that has been in limbo
                if tracking_id not in tracking_history:
                    for existing_id in potential_matches: # search potential matches for re-association  
                        
                        # skip ids that have already been "claimed" by other detections this frame
                        if existing_id in reclaimed_this_frame: 
                            continue 

                        # id has not been reclaimed this frame, get potential match info from tracking history
                        item = tracking_history[existing_id]

                        # check if current detection class name = the potential match class name
                        if item["name"] == class_name:
                            old_x, old_y = item["center"] # get last known coordinates from potential match
                            dist = math.sqrt((new_x - old_x)**2 + (new_y - old_y)**2) # calculate distance between current detection and potential match

                            # if current detection is within the reassociation radius of a potential match
                            if dist < REASSOCIATION_RADIUS:

                                # if potential match is missing, it has been "reclaimed" so print debug statement ( dont want it happening each frame, only when it finds something lost)
                                if item["missed_count"] > 0:
                                    print(f"  [RECLAIM] Found Lost ID {existing_id} at dist {dist:.1f}px (Replaced AI {ai_id})")
                                
                                id_map[ai_id] = existing_id # assign potential match id as the value to the id map, using the orginal ai id given at detection time as the key
                                tracking_id = existing_id # assign the found match id to the current tracking id 
                                reclaimed_this_frame.add(existing_id) # add found id to list for reclaimed items this frame
                                break
                
                ## STATE & RECORD DEBUG ##
                # append current detection id to list of current detections in this frame
                current_frame_detections.append(tracking_id)
                
                # current detection is new, create new item in tracking history
                if tracking_id not in tracking_history: 
                    print(f"  [NEW] Initializing ID {tracking_id} ({class_name})")
                    tracking_history[tracking_id] = {
                        "name": class_name, 
                        "center": (int(new_x), int(new_y)),
                        "seen_count": 1,
                        "missed_count": 0
                    }
                # current detection is old, update tracking history item info      
                else:
                    item = tracking_history[tracking_id]
                    old_x, old_y = item["center"]
                    distance = math.sqrt((new_x - old_x)**2 + (new_y - old_y)**2) # calculate distance moved to check if it is moving
                    item["center"] = (int(new_x), int(new_y))
                    item["seen_count"] += 1
                    item["missed_count"] = 0
                    
                    # if item has been seen for enough frames to be considered "real", set its state using the distance calculated.  
                    if item["seen_count"] == PERSISTENCE_THRESHOLD:
                        state = "MOVING" if distance > MOVEMENT_THRESHOLD else "STATIONARY"
                        print(f"  [STATE] {tracking_id} is now {state} (Dist: {distance:.1f}px)")

                ## CONSTRUCT BOUNDING BOX ## 
                tx, ty = int(new_x - new_w/2), int(new_y - new_h/2) # calculate top corners
                bx, by = int(new_x + new_w/2), int(new_y + new_h/2) # calculate bottom corners
                color = (0, 255, 0) if tracking_id in reclaimed_this_frame or tracking_id in potential_matches else (255, 255, 0) # set colour
                cv2.rectangle(frame, (tx, ty), (bx, by), color, 2) # draw bounding box
                cv2.putText(frame, f"ID {tracking_id} ({conf:.2f})", (tx, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) # draw detection info text

        ## LIMBO & DROPOUT ##
        missing_ids = [tracking_id for tracking_id in tracking_history if tracking_id not in current_frame_detections] # list of ids in tracking history but not seen in current frame 

        # if item is missing (limbo or dropped out), update counter for "frames gone without detection"
        for tracking_id in missing_ids:
            item = tracking_history[tracking_id]
            item["missed_count"] += 1
            
            # item is in limbo if it is missing
            if item["missed_count"] == 1:
                print(f"  [LOST] ID {tracking_id} entered Limbo.")

            # if item has not been detected long enough, remove item from tracking history and id alias map
            if item["missed_count"] >= DROPOUT_THRESHOLD:
                print(f"  [DROPOUT] ID {tracking_id} removed after {DROPOUT_THRESHOLD} frames.")
                del tracking_history[tracking_id]
                keys_to_remove = [k for k, v in id_map.items() if v == tracking_id]
                for k in keys_to_remove: 
                    del id_map[k]
                break 

        ## CONSTRUCT HUD OVERLAY ##
        hud_overlay = frame.copy() # create duplicate frame as a "scratchpad"
        hud_height = 40 + (len(tracking_history) * 30) # calculate dynamic overlay height
        cv2.rectangle(hud_overlay, (0, 0), (450, hud_height), (0, 0, 0), -1) # draw hud overlay rectangle
        frame = cv2.addWeighted(hud_overlay, 0.4, frame, 0.6, 0) # merge hud overlay with current frame 
        y_offset = 30 # set vertical offset of hud overlay 

        # show tracked item info and update hud overlay height
        for tracking_id, item in tracking_history.items():
            status = f"LOST({item['missed_count']})" if tracking_id in missing_ids else "LIVE"
            cv2.putText(frame, f"ID {tracking_id}: {item['name']} | {status}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) # draw hud text
            y_offset += 30

        cv2.imshow("Savor tracking - Live Test", frame) # open live feed window

        # break while loop when "q" is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    # close camera and live feed window
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()