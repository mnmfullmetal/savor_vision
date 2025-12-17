from ultralytics import YOLO
import numpy
import cv2
from collections import defaultdict
from utils import find_model_path

# --- STEP 2: THE RULES (CONSTANTS) ---
# Define the path to your trained .pt file
model_path = find_model_path()
# Define the CONFIDENCE_THRESHOLD (e.g., 0.6) -> "Must be 60% sure"
# Define the PERSISTENCE_THRESHOLD (e.g., 10 frames) -> "Must exist for 0.3s"
# Define the DROPOUT_TOLERANCE (e.g., 5 frames) -> "Can disappear for 0.15s"

# --- STEP 3: THE MEMORY (STATE) ---
# Create a dictionary to remember: "How many frames have I SEEN this ID?"
# Create a dictionary to remember: "How many frames has this ID been MISSING?"

# --- STEP 4: THE SETUP ---
    # Load the AI model from the path
    # Open the connection to the Webcam
    # Set the camera resolution to 1280x720 (High Def)

    # --- STEP 5: THE ENGINE (LOOP) ---
    # Start an infinite loop that runs forever
        # 1. Read one single frame from the camera
        #    If reading fails, break the loop

        # 2. INJECT THE BRAIN (Inference)
        #    Run model.track() on the frame
        #    Enable 'persist=True' so it remembers IDs
        #    Use our CONFIDENCE_THRESHOLD

        # 3. UNPACK THE DATA
        #    Check: Did the AI see anything at all? (Is results not None?)
        #    If YES:
        #       Get the list of Boxes (x, y coordinates)
        #       Get the list of IDs (Who is who?)
        #       Get the list of Class Names (Milo vs Salt)
        
        #       Create a list of "current_ids" that we saw RIGHT NOW

        #       --- STEP 6: THE FILTER (Logic Loop) ---
        #       Loop through every object the AI found THIS FRAME:
        #           Reset the "Missing" counter for this ID to 0 (It's back!)
        #           Add +1 to the "Seen" counter for this ID
        
        #           Check: Is "Seen" counter > PERSISTENCE_THRESHOLD?
        #               If YES: 
        #                   Status = "CONFIRMED"
        #                   Set Color = Green
        #               If NO:
        #                   Status = "SCANNING"
        #                   Set Color = Red

        #           --- STEP 7: THE VISUALIZATION (UI) ---
        #           Draw the Rectangle using the specific Color
        #           Create a Label string (e.g., "Milo [STOCK]")

