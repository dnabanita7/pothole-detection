import streamlit as st
import cv2
import tempfile
import numpy as np
import supervision as sv
from roboflow import Roboflow
import os

# -------------------------------
# 1. PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Pothole Detector", layout="wide")
st.title("Indian Road Pothole Detector")
st.markdown("Upload a photo or a short video clip to detect and count potholes.")

# -------------------------------
# 2. SIDEBAR SETTINGS
# -------------------------------
with st.sidebar:
    st.header("Configuration")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)

    # Move this to env variable in production
    API_KEY = "MozIyzBJ7nMRkP0x2EmF"
    WORKSPACE_ID = "nabanitas-workspace-urt3p"
    PROJECT_ID = "pothole-detection-using-yolov8-fsf2h"
    VERSION = 1

# -------------------------------
# 3. LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
    return project.version(VERSION).model

model = load_model()

# -------------------------------
# 4. FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader(
    "Choose an image or video...",
    type=["jpg", "jpeg", "png", "mp4", "mov"]
)

if uploaded_file is not None:

    is_video = uploaded_file.name.lower().endswith(('.mp4', '.mov'))

    # ===============================
    # IMAGE PROCESSING
    # ===============================
    if not is_video:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        with st.spinner('Detecting potholes...'):

            result = model.predict(image, confidence=conf_threshold * 100).json()
            detections = sv.Detections.from_inference(result)

            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()

            annotated_image = box_annotator.annotate(
                scene=image.copy(),
                detections=detections
            )
            annotated_image = label_annotator.annotate(
                scene=annotated_image,
                detections=detections
            )

            col1, col2 = st.columns(2)

            with col1:
                st.image(image, channels="BGR", caption="Original Image")

            with col2:
                st.image(annotated_image, channels="BGR", caption="Detections")

            st.metric("Potholes Detected", len(detections))

    # ===============================
    # VIDEO PROCESSING
    # ===============================
    else:
        #  Ensure proper extension
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()

        video_info = sv.VideoInfo.from_video_path(tfile.name)

        byte_tracker = sv.ByteTrack()
        box_annotator = sv.BoxAnnotator()

        unique_pothole_ids = set()
        raw_output_path = "pothole_output.mp4"
        final_output_path = "fixed_output.mp4"

        st.info("Processing video frames. This may take a minute...")
        progress_bar = st.progress(0)

        def callback(frame, index):
            progress_bar.progress((index + 1) / video_info.total_frames)

            result = model.predict(frame, confidence=conf_threshold * 100).json()
            detections = sv.Detections.from_inference(result)

            detections = byte_tracker.update_with_detections(detections)

            if detections.tracker_id is not None:
                for tid in detections.tracker_id:
                    unique_pothole_ids.add(tid)

            return box_annotator.annotate(
                scene=frame.copy(),
                detections=detections
            )

        # Run processing
        sv.process_video(
            source_path=tfile.name,
            target_path=raw_output_path,
            callback=callback
        )

        # Force browser-compatible codec
        st.info("Encoding video for playback...")
        os.system(
            f"ffmpeg -y -i {raw_output_path} -vcodec libx264 -acodec aac {final_output_path}"
        )

        # Read video as bytes (MIME fix)
        st.success("Video processing complete!")

        with open(final_output_path, "rb") as video_file:
            video_bytes = video_file.read()
            st.video(video_bytes)

        st.metric("Total Unique Potholes Found", len(unique_pothole_ids))
