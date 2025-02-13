import os
import logging
from pathlib import Path
from typing import NamedTuple
import cv2
import numpy as np
import streamlit as st
from twilio.rest import Client
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sample_utils.download import download_file

st.set_page_config(
    page_title="Image Detection",
    page_icon="\U0001F4F7",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

MODEL_URL = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"
MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"
download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)

if "twilio" in st.secrets:
    TWILIO_ACCOUNT_SID = st.secrets["twilio"].get("account_sid", "")
    TWILIO_AUTH_TOKEN = st.secrets["twilio"].get("auth_token", "")
    TWILIO_PHONE_NUMBER = st.secrets["twilio"].get("from_phone", "")
    TWILIO_ENABLED = all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER])
else:
    st.error("Twilio secrets not found! Check your secrets.toml configuration.")
    TWILIO_ENABLED = False

user_phone_number = st.text_input("Enter your phone number (with country code):", "")

def send_sms_alert(user_number):
    """Send an SMS alert when damage is detected."""
    if not TWILIO_ENABLED or not user_number:
        st.warning("Twilio is not configured, or phone number is missing!")
        return
    
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body="Road damage detected! Please take necessary action.",
            from_=TWILIO_PHONE_NUMBER,
            to=user_number
        )
        logger.info(f"SMS alert sent to {user_number}: {message.sid}")
    except Exception as e:
        logger.error(f"Failed to send SMS: {e}")
        st.error(f"Failed to send SMS: {e}")

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

st.title("Road Damage Detection - Image")
st.write("Detect the road damage using an image. Upload the image and start detecting.")

image_file = st.file_uploader("Upload Image", type=['png', 'jpg'])

score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

if image_file is not None:
    image = Image.open(image_file)
    col1, col2 = st.columns(2)

    _image = np.array(image)
    h_ori, w_ori = _image.shape[:2]

    image_resized = cv2.resize(_image, (640, 640), interpolation=cv2.INTER_AREA)
    results = net.predict(image_resized, conf=score_threshold)

    detections = []
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

    if detections and user_phone_number:
        send_sms_alert(user_phone_number)

    annotated_frame = results[0].plot()
    _image_pred = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation=cv2.INTER_AREA)

    with col1:
        st.write("#### Image")
        st.image(_image)

    with col2:
        st.write("#### Predictions")
        st.image(_image_pred)

        buffer = BytesIO()
        _downloadImages = Image.fromarray(_image_pred)
        _downloadImages.save(buffer, format="PNG")
        _downloadImagesByte = buffer.getvalue()

        st.download_button(
            label="Download Prediction Image",
            data=_downloadImagesByte,
            file_name="RDD_Prediction.png",
            mime="image/png"
        )

        if detections:
            pdf_report = generate_pdf_report(detections)
            st.download_button(
                label="Download Report (PDF)",
                data=pdf_report,
                file_name="Road_Damage_Report.pdf",
                mime="application/pdf"
            )

def generate_pdf_report(detections):
    """Generate a PDF report for detected road damage."""
    pdf_buffer = BytesIO()
    pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
    pdf.setTitle("Road Damage Detection Report")

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(200, 750, "Road Damage Detection Report")
    pdf.setFont("Helvetica", 12)

    y_position = 720
    pdf.drawString(50, y_position, "Detected Cracks:")
    y_position -= 20

    for idx, det in enumerate(detections, start=1):
        text = f"{idx}. {det.label} | Score: {det.score:.2f} | Box: {det.box.tolist()}"
        pdf.drawString(50, y_position, text)
        y_position -= 20
        if y_position < 50:
            pdf.showPage()
            pdf.setFont("Helvetica", 12)
            y_position = 750

    pdf.save()
    pdf_buffer.seek(0)
    return pdf_buffer
