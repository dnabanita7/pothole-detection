import cv2
import supervision as sv
from roboflow import Roboflow

# 1. Initialize the Model
rf = Roboflow(api_key="MozIyzBJ7nMRkP0x2EmF")
project = rf.workspace("pothole-detector-dmdod").project("pothole-clzln")
model = project.version(1).model

# 2. Setup Annotators (for drawing boxes)
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# --- FOR IMAGES ---
def process_image(path):
    image = cv2.imread(path)
    result = model.predict(path, confidence=40).json()
    detections = sv.Detections.from_inference(result)
    
    annotated_image = box_annotator.annotate(scene=image, detections=detections)
    cv2.imwrite("pothole_result.jpg", annotated_image)
    print("Image saved as pothole_result.jpg")

# --- FOR VIDEOS ---
def process_video(path):
    # Initialize Tracker to avoid double-counting
    byte_tracker = sv.ByteTrack()
    
    def callback(frame, index):
        result = model.predict(frame, confidence=40).json()
        detections = sv.Detections.from_inference(result)
        detections = byte_tracker.update_with_detections(detections)
        
        annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
        return label_annotator.annotate(scene=annotated_frame, detections=detections)

    sv.process_video(source_path=path, target_path="pothole_video_result.mp4", callback=callback)
    print("Video saved as pothole_video_result.mp4")

# Change these to your actual file names
process_image("Pothole_Image_Data/4.jpg")
# process_video("road_video.mp4")
