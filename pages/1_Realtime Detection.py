import logging
import queue
from pathlib import Path
from typing import List, NamedTuple
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from ultralytics import YOLO
from twilio.rest import Client
from sample_utils.download import download_file
from sample_utils.get_STUNServer import getSTUNServer

st.set_page_config(
    page_title="Realtime Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

logger = logging.getLogger(__name__)
MODEL_URL = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"
MODEL_LOCAL_PATH = Path("./models/YOLOv8_Small_RDD.pt")

download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)

STUN_STRING = "stun:" + str(getSTUNServer())
STUN_SERVER = [{"urls": [STUN_STRING]}]

cache_key = "yolov8smallrdd"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = YOLO(MODEL_LOCAL_PATH)
    st.session_state[cache_key] = net

CLASSES = ["Longitudinal Crack", "Transverse Crack", "Alligator Crack", "Potholes"]

# Load Twilio credentials
twilio_enabled = "twilio" in st.secrets
if twilio_enabled:
    TWILIO_ACCOUNT_SID = st.secrets["twilio"].get("account_sid", "")
    TWILIO_AUTH_TOKEN = st.secrets["twilio"].get("auth_token", "")
    TWILIO_PHONE_NUMBER = st.secrets["twilio"].get("from_phone", "")
    twilio_enabled = all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER])
else:
    st.warning("Twilio secrets not found! SMS alerts are disabled.")
    twilio_enabled = False

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

st.title("Road Damage Detection - Realtime")
st.write("Detect road damage in real-time using a webcam for on-site monitoring.")

result_queue = queue.Queue()

user_phone_number = st.text_input("Enter phone number for alerts (optional):")
score_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

def send_sms_alert(damage_type):
    if not twilio_enabled or not user_phone_number.startswith("+"):
        return
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=f"Alert: Detected severe road damage: {damage_type}.",
            from_=TWILIO_PHONE_NUMBER,
            to=user_phone_number
        )
        st.success(f"SMS Sent: {damage_type} alert.")
    except Exception as e:
        st.error(f"Twilio Error: {e}")

severe_damage_counter = {}

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    global severe_damage_counter
    image = frame.to_ndarray(format="bgr24")
    h_ori, w_ori = image.shape[:2]
    image_resized = cv2.resize(image, (640, 640))
    results = net.predict(image_resized, conf=score_threshold)
    
    for result in results:
        boxes = result.boxes.cpu().numpy()
        detections = [
            Detection(
                class_id=int(_box.cls),
                label=CLASSES[int(_box.cls)],
                score=float(_box.conf),
                box=_box.xyxy[0].astype(int),
            ) for _box in boxes
        ]
        result_queue.put(detections)

    annotated_frame = results[0].plot()
    output_frame = cv2.resize(annotated_frame, (w_ori, h_ori))
    
    for detection in result_queue.queue:
        if detection.label in ["Potholes", "Alligator Crack"]:
            severe_damage_counter[detection.label] = severe_damage_counter.get(detection.label, 0) + 1
            if severe_damage_counter[detection.label] >= 5:
                send_sms_alert(detection.label)
                severe_damage_counter[detection.label] = 0
    
    return av.VideoFrame.from_ndarray(output_frame, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="road-damage-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": STUN_SERVER},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": {"width": {"ideal": 1280, "min": 800}}, "audio": False},
    async_processing=True,
)

st.divider()

if st.checkbox("Show Predictions Table", value=False):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        while True:
            result = result_queue.get()
            labels_placeholder.table(result)
