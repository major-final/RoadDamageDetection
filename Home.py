import streamlit as st

st.set_page_config(
    page_title="Road Damage Detection Apps",
    page_icon="🛣️",
)


st.divider()
st.title("Road Damage Detection Application")


st.markdown(
    """
    Introducing our Road Damage Detection Apps, powered by the YOLOv8 deep learning model trained on Crowdsensing-based Road Damage Detection Challenge 2022 Dataset.
    
    This application is designed to enhance road safety and infrastructure maintenance by swiftly identifying and categorizing various forms of road damage, such as potholes and cracks.

    There are four types of damage that this model can detect:
    - Longitudinal Crack
    - Transverse Crack
    - Alligator Crack
    - Potholes

    The model is trained on the YOLOv8 small model using Japan and India CRDDC2022 dataset.

    You can select the app from the sidebar to try and experiment with different inputs **(real-time webcam, video, and images)** depending on your use case.

    #### Documentation and Links
    - GitHub Project Page [GitHub](https://github.com/oracl4/RoadDamageDetection)
    - Contact: it.mahdi.yusuf@gmail.com

    #### License and Citations
    - Road Damage Dataset from Crowdsensing-based Road Damage Detection Challenge (CRDDC2022)
    - All rights reserved under YOLOv8 license permitted by [Ultralytics](https://github.com/ultralytics/ultralytics) and the [Streamlit](https://streamlit.io/) framework
"""
)

st.divider()

st.markdown(
    """
    This project was created for the [Road Damage Detection Challenge](https://s.id/RDDHariJalan23) by the [Ministry of Public Works and Housing](https://pu.go.id/) in celebration of "Peringatan Hari Jalan 2023".
    """
)
