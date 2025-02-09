import os
import logging
from pathlib import Path
from typing import List, NamedTuple

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from twilio.rest import Client  # Twilio for SMS notifications

from sample_utils.download import download_file

st.set_page_config(
    page_title="Video Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Fixing Path Issues in Streamlit
HERE = Path(".").resolve()
ROOT = HERE.parent

logger = logging.getLogger(__name__)

MODEL_URL = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"
MODEL_LOCAL_PATH = ROOT / "models/YOLOv8_Small_RDD.pt"
download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)

# Load Twilio credentials from Streamlit Secrets
TWILIO_ACCOUNT_SID = st.secrets["twilio"]["account_sid"]
TWILIO_AUTH_TOKEN = st.secrets["twilio"]["auth_token"]
TWILIO_MESSAGING_SERVICE_SID = st.secrets["twilio"]["messaging_service_sid"]
TO_PHONE_NUMBER = st.secrets["twilio"]["to_phone"]
FROM_PHONE_NUMBER = st.secrets["twilio"]["from_phone"]

def send_notification(damage_type):
    """Send an SMS notification when a crack is detected."""
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    message = client.messages.create(
        body=f"Alert: {damage_type} detected on the road! Immediate action may be required.",
        from_=FROM_PHONE_NUMBER,
        to=TO_PHONE_NUMBER
    )
    print(f"Notification sent: {message.sid}")

def write_bytesio_to_file(file_path, file_bytes):
    """Writes an uploaded file (BytesIO) to disk."""
    with open(file_path, "wb") as out_file:
        out_file.write(file_bytes.read())

# Load YOLO model with session-specific caching
cache_key = "yolov8smallrdd"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = YOLO(str(MODEL_LOCAL_PATH))
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

# Ensure temp directory exists
TEMP_DIR = Path("./temp")
TEMP_DIR.mkdir(exist_ok=True)

temp_file_input = TEMP_DIR / "video_input.mp4"
temp_file_infer = TEMP_DIR / "video_infer.mp4"

def process_video(video_file, score_threshold):
    """Processes uploaded video, detects road damage, and sends notifications."""
    write_bytesio_to_file(temp_file_input, video_file)
    video_capture = cv2.VideoCapture(str(temp_file_input))

    if not video_capture.isOpened():
        st.error('Error opening the video file')
        return

    _width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    _height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _fps = video_capture.get(cv2.CAP_PROP_FPS)
    _frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    inference_bar = st.progress(0, text="Processing video...")
    image_location = st.empty()

    fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
    cv2_writer = cv2.VideoWriter(str(temp_file_infer), fourcc_mp4, _fps, (_width, _height))

    _frame_counter = 0
    damage_detected = False

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = net.predict(frame, conf=score_threshold)
        annotated_frame = results[0].plot()

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for _box in boxes:
                detection = Detection(
                    class_id=int(_box.cls),
                    label=CLASSES[int(_box.cls)],
                    score=float(_box.conf),
                    box=_box.xyxy[0].astype(int),
                )
                if detection.score > score_threshold:
                    damage_detected = True
                    send_notification(detection.label)

        _image_pred = cv2.resize(annotated_frame, (_width, _height), interpolation=cv2.INTER_AREA)
        cv2_writer.write(cv2.cvtColor(_image_pred, cv2.COLOR_RGB2BGR))
        image_location.image(_image_pred)
        _frame_counter += 1
        inference_bar.progress(_frame_counter / _frame_count, text="Processing video...")

    inference_bar.empty()
    video_capture.release()
    cv2_writer.release()

    st.success("Video Processed!")
    with open(temp_file_infer, "rb") as f:
        st.download_button("Download Processed Video", data=f, file_name="RDD_Prediction.mp4", mime="video/mp4")

st.title("Road Damage Detection - Video")
st.write("Upload a video to detect road damage and receive notifications for detected issues.")

video_file = st.file_uploader("Upload Video", type=["mp4"], disabled=False)
score_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

if video_file and st.button("Process Video"):
    st.warning(f"Processing Video: {video_file.name}")
    process_video(video_file, score_threshold)
