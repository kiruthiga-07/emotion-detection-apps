import streamlit as st
import cv2
from fer import FER
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# --- Setup & Caching ---
st.set_page_config(page_title="Real-Time Emotion AI", layout="centered")

@st.cache_resource
def get_detector():
    # mtcnn=False is much faster for live video
    return FER(mtcnn=False)

detector = get_detector()

# --- Real-Time Video Logic ---
class EmotionProcessor(VideoTransformerBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Detect emotion on every frame
        emotions = detector.detect_emotions(img)
        
        for face in emotions:
            (x, y, w, h) = face["box"]
            dominant = max(face["emotions"], key=face["emotions"].get)
            
            # Draw on the live frame
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, dominant, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- App Interface ---
st.title("🎭 Emotion Detection App")

option = st.sidebar.selectbox("Mode", ["Live Webcam", "Upload Image"])

if option == "Live Webcam":
    st.subheader("Live Real-Time Detection")
    st.write("Click 'Start' below to turn on your camera.")
    
    # This replaces the camera_input for real-time video
    webrtc_streamer(
        key="emotion-live",
        video_processor_factory=EmotionProcessor,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )

else:
    uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        # Static detection logic
        results = detector.detect_emotions(img_array)
        # ... (same drawing logic as before)
        st.image(img_array, caption="Detected Emotion")
