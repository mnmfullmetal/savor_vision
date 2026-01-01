
# Savor Vision Dev Log #01
**Project Name:** Savor - Vision  
**Current Milestone:** Annotated Dataset/ First Trained Model  
**Primary Engine:** YOLOv8 (Hardware Target: PC Development for Raspberry Pi 5 Deployment w/ Hailo-8 AI Hat+)  
**Toolchain:** Roboflow (Annotation & Dataset Creation), OBS (Video Capture), OpenCV (Frame manipulation), Python 3.10 (Logic)

## 1. Project Evolution
**Change:** Collection and annotation (labelling) of a small scale dataset and the training of my first computer vision model.
**Why:** To teach YOLO model how to detect/track a small collection (3) of pantry items, while simultaneously teaching myself how to train and use AI computer vision models. 

## 2. Developmental Phases
### Phase I: Dataset Collection
The computer vision or "CV" model (YOLOv8) is a newborn ready for me to shape its knowledge base and so the first major step in my journey was to gather valuable data, a collection of images for the baby CV model to learn from. 
* **Why:** An untrained CV model is a blank slate that has to be taught how to detect the items I need it track. In order to do this it needs to be taught using a variety of images that picture the objects that need to be tracked aswell as ones that done (null dataset). 
* **Desired Outcome:** A collection of small yet valuable dataset images that I can use `Roboflow` to programatically change for more variety and eventually train my first model.
* **Result & Pivot:** Successful result! No need to pivot. I used 2, 16mp cameras and my laptop with `OBS` to capture videos of 3 pantry items (milo tin, green tea box, table salt bottle) of different shapes on a faux pantry shelf in my room. I specifically chose the milo tin and green tea box because their colours were close, in hopes that it will teach the idea that 2 objects can be the same colour but not the same object. The table salt bottle had contrasting colours when compared to the other 2 and an odd tapered shape and that is why I chose it. I shot from 2 angles, front on (**Cam A**) with a downwards angle of aprox 45 degrees and a distance of 50cm from the shelf top. The 2nd angle (**Cam B**) was looking at the shelf diagonally from the right of **Cam A**'s view. **Cam B** was also aprox 5-10 cm shorter than Cam A and therefore did not have the same downwards angle, which was closer to 30 degrees. I shot 3 different style scenes for each camera angle:  
* **The Perfect shot:** On **Cam A** All items lined up in a row, spread apart and easy to distinguish, and rotated them at intervals. This gave me my "perfect" still shots front a front view. On **Cam B**, because of the diagonal viewpoint and lowered sightline, the perfect shots for this had to be spread but insteasd of horizontally acrosss the breadth of the faux pantry, they spread deeper into the pantry.
* **The Crowded shot:** On **Cam A** I pushed my items close together so they were touching one another, **Cam B** I had them cascading away from the camera so that parts of the back two were occluded from view.
* **The Action shot:** For both camera angles, I shot myself reaching in from out of frame and taking one item at a time, then putting them back. I did this with both "crowded" and "perfect" layouts of items.
I then created a new object detection project on `Roboflow` and from these videos I created a set of dataset images by using `Roboflow` to extract frames at intervals.
### Phase II: Teaching A Digital Child What Milo Is (Annotation & Training Mode)  
* **Why:** Now that I have my dataset, I have to **annotate** the data, "label" it in `Roboflow`. This requires drawing polygons around each item, hugging the pixels as closely as possible in order to teach the model how to "see" the objects I want it to, and when that is done I can use `Roboflow` to create a **version** (collection of programatically changed images and instructions) to then train my model with.
* **Desired Outcome:** An annotated dataset and my first working model.
* **Result & Pivot:** Success! I created an object detection project in `Roboflow` and  After many hours manually annotating the dataset I created my first version and downloaded it to my PC to train my model locally. In order to train the model I had to construct a python script to run the training using my RTX 4080 GPU and on the first run through it had an averge of 0.93 confidence across the board. While this is a major success, the high score of confidence doesnt mean much until it becomes more intelligent.
### Phase III: First Test Cycle (Predict Mode)  
* **Why:** Before moving into complex tracking logic, I wanted to establish a stable foundation by testing my newly trained model thoroughly.
* **Desired Outcome:** Achieving a stable **Predict** (method from `ultralytics` library belonging to the YOLO class) result on still images and a live camera feed, proving the training results.
* **Result & Pivot:** The first round of testing on still images pulled from the raw data video seemed promising, though it had a few inaccuracies such as missing a clearly visible Milo tin, or mistaking the Milo for the green tea box at certain angles due to similar colourss. I proceeded to move onto testing **Predict** on a live camera feed and this is where it all fell apart. Detections were unstable; for example, the model was "hallucinating" the salt bottle on the shirt which shared similar colours and the model would lose confidence at certain angles. To pivot I included negative samples (images of the room/shirt with no labels) and varied angles, annotated them and ran **Predict** on a live camera feed test and while not perfect, it was far more stable than previously, making a successful pivot and proof that I was heading in the correct direction.

## 3. Challenges & Outcomes
* **Challenge :** Learning a new toolset. I had never used `Roboflow` or `OpenCV`,  nor had I ever trained a computer vision model before so nearly all aspects of what I was undertaking was new (albiet exciting) to me.    
**Outcome:** I feel far more confident now that I have a basic understanding of the workflow/pipeline and how to use the tools within it. My largest fear was that I would spend a lot of time manually annotating only for it to be completely useless because I had done it wrong, and while I did spend a decent amount of time manually labelling images, my hardwork seems to have paid off.  
* **Challenge:** Teaching the model that not all collections of pixels with the same colours are equal. The model was creating phantom detections due to the lack of null dataset images     
**Outcome:** After making the dataset more robust it has to look "deeper" because it may just be my shirt that has the same colour. In my research into the solution, I came across a good analogy."The AI is like a newborn that you have taught how to identify a dog, then it sees a wolf and thinks it is a dog. It needs to be taught that not all hairy, four legged creatures are dogs". I had taught it what a positive looks like but not what a negative looks like (not well atleast).
  Overall, I was successful in teaching the model that when it sees a collection of pixels that are a similar colour to an object it is looking for by adding null data to the dataset and it no longer created those phantom detections with high confidence.

## 4. Summary of Milestone
I have successfully trained my first model to detect a small group of items in a a single frame with a high level of confidence while they remain stationary. 

## 5. Next Steps
Begin testing with **Track** (method from `ultralytics` library belonging to the YOLO class) instead of **Predict** and attempt to "track" the objects through space/time. Currently the model has no memory. In every frame it is given, the model finds the objects all over again and it can not track their movements through space/time which I will need it to do for pantry management within  Savor.



<br>
<br>



# Savor Vision Dev Log #02
**Project Name:** Savor - Vision  
**Current Milestone:** Logic-Stable Prototype / Dataset Pivot  
**Primary Engine:** YOLOv8 (Fixed Hardware Target: PC Development for Raspberry Pi 5 Deployment w/ Hailo-8 AI Hat+)  
**Toolchain:** Roboflow (Annotation & Dataset Creation), OpenCV (Frame manipulation), Python 3.10 (Logic)

## 1. Project Evolution
**Change:** Transition from Single-Frame Detection to Multi-Frame Tracking Logic.  
**Why:** Detection alone provides "what" and "where" in a static instant, but pantry management requires "who" over time. To maintain an accurate inventory, the system must recognise that a tin at coordinate A is the same physical object that was at coordinate B a second ago.

## 2. Developmental Phases
### Phase I: Object Tracking Test (Track Mode)
* **Why:** To test the accuracy of the trackers (`ByteTrack` & `BotSort`) inherently provided by the YOLO model and begin detecting **and** tracking items in order to implement **Savor** level logic for inventory (database) management. 
* **Desired Outcome:** A model capable of finding and tracking items, applying ID's to the items and preserving them in memory as they move through the scene.
* **Result & Pivot:** Pivot Required. The model suffered from detection fragmentation and created phantom detections. For example, whenever a hand partially covered an object, the bounding boxes drawn by the model split into multiple separate objects, breaking the tracker's ability to maintain a single ID, creating multiple false ID's for the same item. When an item was moving through the scene, the model was creating "ghost" detections with ID's, again creating multiple false ID's for the same item. When the real item is found again in both scenerios, it is assigned a new ID, not its original, proving that the model with the trackers are not yet capable of tracking the object in scenarios I need it to. To pivot, I fine-tuned the models inbuilt trackers and tested between both `BotSort` and `ByteTrack` in an attempt to gain more consitent accuracy. 
### Phase II: The "Master Record" Integration
* **Why:** My pivot in the previous phase only worked to confuse me more, as it seemed that no matter what I tried on the tracker side I could not get the tracking to work accurately and as intended. The problems from the previous phase persisted at different intensities depending on the tracker configuration. To counter this and simultaneously start building the app-level logic, I need to create a logical "referee" for the AI's raw tracking data to try and keep consistent ID's. 
* **Desired Outcome:** Eliminate "Double Vision" and "Identity Flipping" where objects change IDs mid-movement or due to occlusion.
* **Result & Pivot:** Pivot required. I tried implementing my own **spatial re-association** and **overlap suppression** logic, I created a "Ghost" memory that holds lost IDs for 300 frames (10 seconds), everything I tried never achieved the desired outcome of stable item persistence. None of it made sense to me because by all rights with the trackers and my own tracking logic working together the model should not have been acting as it was. I began researching from the ground up and found the issue nearly immediately. I had annotated the dataset using pixel-perfect masks, which is ideal for **intance segmentation** but I had created an **object detection** project in `Roboflow`. This explained what was happening with the fragmentation and flipping of identities, the model was turning my pixel perfect data into 2D bounding boxes that would skew while moving and fragment when occlusion seperated the item in half (it was creating 2 bounding boxes instead of remembering the objects centre point was behind the occlusion).  
To pivot, I shifted the training strategy from "visible-only" to "expected-whole" (**The Envelope Rule**).
### Phase III: The "Envelope Rule" Pivot
* **Why:** To address phantom detections caused by the Kalman Filter panicking when bounding boxes changed size drastically during occlusion or movement. 
* **Desired Outcome:** A model with a tracker that maintains a constant box size and center-of-mass regardless of how much of the object is hidden.
* **Result & Pivot:** Success! I created an instance segementation project in `Roboflow` and migrated the previously made pixel-perfect dataset over to the new project, trained, tested it and it worked for still, crowded scenes as expected. Then using the same raw dataset I used previously, I annotated each image by drawing 2D bounding boxes of the expected whole of the item, instead of drawing a mask over the only-visible pixels. This allows the math of the tracker to remain stable during movement by keeping the expected center stable. I then trained and tested the new tracking model with the now appropriate dataset and it was a success! The new tracking model was far more stable now it was using a properly annotated dataset for its usecase though it still needed perfecting because it would lose confidence when the items were at certain angles or I moved the item to fast. Despite the issues, the results were promising as it meant I was again heading down the correct path. In doing this I have created a "2 brain system", where the instance segmentation model would be the "audit" brain handling still, crowded items, while the object detection model would handle the "tracking" brain handling items that are moving through the scene.   

## 3. Challenges Faced and Outcomes
* **Challenge:** The model could not accurately track items once they were moving and/or occluded despite trying different trackers and their configurations. Even after going as far as implementing an additional logic layer as referee to try and keep the ID's on track, the model was not tracking as expected.
* **Outcome:** After more research into the issue and discovering that the dataset was annotated using the wrong method, I created 2 different models, one for tracking (object detection) the other for auditing (instance segmentation), with the aim of having the two models work together in unison.

## 4. Summary of Milestone
Despite many issues and pivots along the way, I have successfully moved from unstable static image detection (**Predict**) to stable tracking in dynamic scenes (**Track**). Additionaly, as a result of the many pivots taken, I now have 2 models that each specialise in different aspects of computer vision that can be used together to create a fully capable system where panntry items can be tracked with confidence despite occlusions or movement.

## 5. Next Steps
Continue to test and perfect each model at their respective tasks before moving onto bringing both brains together. 


<br>
<br>

# Savor Vision Dev Log #03
**Project Name:** Savor - Vision
**Current Milestone:** Dual-Brain Architecture / Model Pivot
**Primary Engine:** YOLOv8 (Hardware Target: PC Development for Raspberry Pi 5 Deployment w/ Hailo-8 AI Hat+) 
**Toolchain:** OpenCV (Frame manipulation), Python 3.10 (Logic)

## 1. Project Evolution
**Change:** Migration from a "single-brain" model to a "dual-brain" system (The Watcher and the Accountant).  
**Why:** There is a fundamental conflict in pantry vision requirements. High-speed tracking (the Watcher) requires low confidence thresholds to maintain a "grip" on moving items, while inventory auditing (the Accountant) requires high confidence to ensure 100% accuracy. Splitting these allows each model to be optimized for its specific task without compromising the other.

## 2. Developmental Phases
### Phase I: Project Structure Reformation (Modular Hierarchy)
* **Why:** To organise the projects structure to prepare for the dual-brain system and eventual hardware migration to the Raspi5.
* **Desired Outcome:** A modular project structure where tracking logic, audit logic, and utilities (like path-finding) are isolated.
* **Result & Pivot:** Success! Developed `utils.py` for dynamic path-finding and created separate test beds for each model, ensuring the system is ready for hardware migration and project organisation/cleanliness standards are kept.
### Phase II: Perfecting the Brains
* **Why:** To solve the minor tracking issues identified in Phase III of Dev Log #02.
* **Desired Outcome:** Optimised models for both the Watcher and the Accountant so that they don't lose confidence when the items are at weird angles or when they are moving to fast.
* **Result & Pivot:** Pivot, not due to it being required, but because I found additional information that changes my approach completely. While I was looking through the documentation for `ultralytics` I found "Instance Segementation WITH Object Tracking", capabilities of both the Watcher and the Accountant in the 1 model instead of 2 seperate ones. While this could be done with `yolov8s-seg`, my research has convinced me to make the largest pivot yet and move to `yolov11s-seg` for this unified brain, providing faster and more advanced segmentation and spatial tracking capabilities. 

## 3. Challenges Faced and Outcomes
* **Challenge:"** Reorganising the project structure cleanly and safely.
* **Outcome:** Reorganising the project resulted in no loss of data and provided me with an organised and decoupled development environment ready for migration to the Raspi5.

## 4. Summary of Milestone
While I started out attempting to create and perfect a dual-brain system, with the new information that both segmentation and tracking can be done by the same model I believe now that is the best course of action. Not only that, I have opted for migrating from using YOLOv8 to YOLOv11 for more its more advanced capabilties.

## 5. Next Steps
To train a new model (`yolov11s-seg`) to handle both instance segmentation and object tracking. 



<br>
<br>



# Savor Vision Dev Log #04
**Project Name:** Savor - Vision
**Current Milestone:** Single-Brain, Dual Logic System / HEF File Compilation
**Primary Engine:** YOLOv11 (Hardware Target: PC Development for Raspberry Pi 5 Deployment w/ Hailo-8 AI Hat+) 
**Toolchain:** Roboflow (Annotation & Dataset Creation), OpenCV (Frame manipulation), Python 3.10 (Logic)

## 1. Project Evolution
**Change:** 
**Why:** 

## 2. Developmental Phases
### Phase I: Project Structure Reformation (Modular Hierarchy)
* **Why:** 
* **Desired Outcome:** 
* **Result & Pivot:** 
### Phase II: Perfecting the Brains
* **Why:**
* **Desired Outcome:** 
* **Result & Pivot:** 

## 3. Challenges Faced and Outcomes
* **Challenge:"** 
* **Outcome:** 

## 4. Summary of Milestone

## 5. Next Steps
