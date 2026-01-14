import streamlit as st
import cv2
from fer import FER
import numpy as np
from PIL import Image

# ----------------- Page Setup -----------------
st.set_page_config(page_title="Emotion Detection App", layout="centered")

# Initialize the detector once and cache it for speed
@st.cache_resource
def load_detector():
    # mtcnn=False is MUCH faster for cloud servers
    return FER(mtcnn=False)

detector = load_detector()

# ----------------- Session State -----------------
if "start" not in st.session_state:
    st.session_state.start = False

# ----------------- Welcome Page -----------------
if not st.session_state.start:
    st.title("🎭 Welcome to the Emotion Detection App!")
    st.write("Detect emotions live or from photos.")
    if st.button("Go to Emotion Detection"):
        st.session_state.start = True
        st.rerun()

# ----------------- Main App -----------------
else:
    st.sidebar.title("Settings")
    if st.sidebar.button("Back to Home"):
        st.session_state.start = False
        st.rerun()

    option = st.sidebar.selectbox("Select Input Mode", ("Upload Image", "Use Webcam"))

    def detect_emotion(image):
        result = detector.detect_emotions(image)
        if not result:
            return image, None
        
        for face in result:
            (x, y, w, h) = face["box"]
            dominant_emotion = max(face["emotions"], key=face["emotions"].get)

            # Drawing logic
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, dominant_emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        return image, result

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            image_np = np.array(image.convert('RGB'))
            annotated_image, results = detect_emotion(image_np)
            st.image(annotated_image, channels="RGB")
            if results:
                st.success(f"Detected: {max(results[0]['emotions'], key=results[0]['emotions'].get)}")

    elif option == "Use Webcam":
        st.info("Take a photo to detect your emotion live!")
        img_file_buffer = st.camera_input("Snapshot")
        if img_file_buffer:
            image = Image.open(img_file_buffer)
            image_np = np.array(image.convert('RGB'))
            annotated_image, results = detect_emotion(image_np)
            st.image(annotated_image, channels="RGB")
