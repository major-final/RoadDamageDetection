import os
import logging
from pathlib import Path
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

# Define model path
MODEL_URL = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"
MODEL_LOCAL_PATH = Path("./models/YOLOv8_Small_RDD.pt")

# Download model if it doesn't exist
if not MODEL_LOCAL_PATH.exists():
    download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)

# Load YOLO model
model = YOLO(str(MODEL_LOCAL_PATH))

# Class labels (Make sure this matches your YOLO model's labels)
CLASSES = ["Pothole", "Crack"]  # Modify according to your dataset

# Load Twilio credentials dynamically
if "twilio" in st.secrets:
    TWILIO_ACCOUNT_SID = st.secrets["twilio"].get("account_sid", "")
    TWILIO_AUTH_TOKEN = st.secrets["twilio"].get("auth_token", "")
    TWILIO_PHONE_NUMBER = st.secrets["twilio"].get("from_phone", "")
    TWILIO_ENABLED = all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER])
else:
    st.error("Twilio secrets not found! Check your secrets.toml configuration.")
    TWILIO_ENABLED = False

def send_sms_alert(user_phone_number, damage_type):
    """Send an SMS alert dynamically based on user input."""
    if not TWILIO_ENABLED:
        st.warning("Twilio is not configured correctly. Notifications are disabled.")
        return
    
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=f"Alert: {damage_type} detected on the road! Immediate action may be required.",
            from_=TWILIO_PHONE_NUMBER,
            to=user_phone_number
        )
        logging.info(f"SMS alert sent: {message.sid}")
    except Exception as e:
        logging.error(f"Failed to send SMS: {e}")

# Ensure temp directory exists
TEMP_DIR = Path("./temp")
TEMP_DIR.mkdir(exist_ok=True)

temp_file_input = TEMP_DIR / "video_input.mp4"
temp_file_infer = TEMP_DIR / "video_infer.mp4"

st.title("Road Damage Detection - Video")
st.write("Upload a video to detect road damage and receive notifications for detected issues.")

user_phone_number = st.text_input("Enter your phone number for alerts:")
video_file = st.file_uploader("Upload Video", type=["mp4"], disabled=False)
score_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

def process_video(video_file, score_threshold, user_phone_number):
    """Processes uploaded video, detects road damage, and sends only one notification per video."""
    temp_file_input.write_bytes(video_file.read())
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
    alert_sent = False  # Track if an alert has been sent

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(frame, conf=score_threshold)
        annotated_frame = results[0].plot()

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for _box in boxes:
                class_id = int(_box.cls)
                label = CLASSES[class_id] if class_id < len(CLASSES) else "Unknown"
                score = float(_box.conf)
                
                if score > score_threshold and not alert_sent:
                    send_sms_alert(user_phone_number, label)  # Send notification only once
                    alert_sent = True  # Set flag to avoid further notifications

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

if video_file and user_phone_number and st.button("Process Video"):
    st.warning(f"Processing Video: {video_file.name}")
    process_video(video_file, score_threshold, user_phone_number)
