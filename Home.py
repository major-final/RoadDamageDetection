import streamlit as st

st.set_page_config(
    page_title="Road Damage Detection Apps",
    page_icon="🛣️",
)

st.divider()
st.title("Road Damage Detection Application")

st.markdown(
    """
    Introducing our Road Damage Detection Apps, powered by the YOLOv8 deep learning model trained on the Crowdsensing-based Road Damage Detection Challenge 2022 Dataset.
    
    This application is designed to enhance road safety and infrastructure maintenance by swiftly identifying and categorizing various forms of road damage, such as potholes and cracks.

    There are four types of damage that this model can detect:
    - **Longitudinal Crack**
    - **Transverse Crack**
    - **Alligator Crack**
    - **Potholes**

    The model is trained on the YOLOv8 small model using the Japan and India CRDDC2022 dataset.
    """
)

st.divider()
