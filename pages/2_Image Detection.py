import os
import logging
import time
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import streamlit as st

# Deep learning framework
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

from sample_utils.download import download_file

st.set_page_config(
    page_title="Image Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

MODEL_URL = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"
MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"
download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)

# Session-specific caching
# Load the model
cache_key = "yolov8smallrdd"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = YOLO(MODEL_LOCAL_PATH)
    st.session_state[cache_key] = net

CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

def show_animated_alert(severity):
    alert_placeholder = st.empty()  # Create a placeholder for the alert

    for _ in range(5):  # Blink 5 times
        if severity == "Severe":
            alert_placeholder.error("🚨 SEVERE DAMAGE DETECTED! 🚨")
        elif severity == "Moderate":
            alert_placeholder.warning("⚠️ MODERATE DAMAGE DETECTED! ⚠️")
        time.sleep(0.5)
        alert_placeholder.empty()  # Remove message
        time.sleep(0.5)

    # After blinking, show a final static message
    if severity == "Severe":
        alert_placeholder.error("🚨 SEVERE DAMAGE DETECTED! Take Immediate Action! 🚨")
    elif severity == "Moderate":
        alert_placeholder.warning("⚠️ MODERATE DAMAGE DETECTED! Please be cautious. ⚠️")

st.title("Road Damage Detection - Image")
st.write("Detect the road damage in an image. Upload the image and start detecting. This section can be useful for examining baseline data.")

image_file = st.file_uploader("Upload Image", type=['png', 'jpg'])

score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
st.write("Lower the threshold if there is no damage detected, and increase the threshold if there are false predictions.")

if image_file is not None:
    # Load the image
    image = Image.open(image_file)
    
    col1, col2 = st.columns(2)
    
    # Perform inference
    _image = np.array(image)
    h_ori, w_ori = _image.shape[:2]
    
    image_resized = cv2.resize(_image, (640, 640), interpolation=cv2.INTER_AREA)
    results = net.predict(image_resized, conf=score_threshold)
    
    detections = []  # Store detections
    for result in results:
        boxes = result.boxes.cpu().numpy()
        detections = [
            Detection(
                class_id=int(_box.cls),
                label=CLASSES[int(_box.cls)],
                score=float(_box.conf),
                box=_box.xyxy[0].astype(int),
            )
            for _box in boxes
        ]

    annotated_frame = results[0].plot()
    _image_pred = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation=cv2.INTER_AREA)

    # Original Image
    with col1:
        st.write("#### Image")
        st.image(_image)
    
    # Predicted Image
    with col2:
        st.write("#### Predictions")
        st.image(_image_pred)

        # Determine Severity Level
        if detections:
            max_confidence = max([det.score for det in detections])  # Highest confidence score
            image_area = w_ori * h_ori  # Total image area
        
            # Check if any bounding box is large
            large_box_detected = any(
                ((det.box[2] - det.box[0]) * (det.box[3] - det.box[1])) / image_area > 0.3 for det in detections
            )
        
            # Determine severity
            if max_confidence > 0.6 or large_box_detected:
                severity = "Severe"
            elif max_confidence > 0.4:
                severity = "Moderate"
            else:
                severity = "Mild"
        
            st.write(f"### Severity Level: {severity}")
            
            if severity in ["Moderate", "Severe"]:
                show_animated_alert(severity)  # Show animated alert
        else:
            st.write("### Severity Level: No damage detected")
