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
    # We use mtcnn=False because it is much more stable on Streamlit Cloud servers
    return FER(mtcnn=False)

detector = load_detector()

# ----------------- Session State -----------------
if "start" not in st.session_state:
    st.session_state.start = False

# ----------------- Welcome Page -----------------
if not st.session_state.start:
    st.title("🎭 Welcome to the Emotion Detection App!")
    st.write("""
    This app allows you to:
    - Detect emotions from uploaded images
    - Detect emotions using your webcam
    """)
    if st.button("Go to Emotion Detection"):
        st.session_state.start = True
        st.rerun()

# ----------------- Main App -----------------
else:
    st.title("🎭 Real-Time Emotion Detection App")
    
    # Sidebar back button
    if st.sidebar.button("Back to Home"):
        st.session_state.start = False
        st.rerun()

    option = st.sidebar.selectbox(
        "Select Input Mode",
        ("Upload Image", "Use Webcam")
    )

    def detect_emotion(image):
        # Convert RGB (Streamlit) to BGR (OpenCV) for the detector
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        result = detector.detect_emotions(image_bgr)
        all_faces = []

        if not result:
            return image, None

        for face in result:
            (x, y, w, h) = face["box"]
            emotions = face["emotions"]
            dominant_emotion = max(emotions, key=emotions.get)

            # Draw rectangle and text on the original RGB image
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)
            cv2.putText(image, dominant_emotion, (x, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 3)

            all_faces.append({
                "dominant_emotion": dominant_emotion,
                "emotions": emotions
            })
        return image, all_faces

    # --- Upload Mode ---
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            image_np = np.array(image.convert('RGB'))
            
            with st.spinner('Analyzing...'):
                annotated_image, faces = detect_emotion(image_np)
            
            st.image(annotated_image, channels="RGB", caption="Processed Image")

            if faces:
                st.success("Face Detected!")
                for i, face in enumerate(faces):
                    st.write(f"**Face {i+1}: {face['dominant_emotion']}**")
            else:
                st.warning("No faces detected. Try a clearer photo.")

    # --- Webcam Mode ---
    elif option == "Use Webcam":
        img_file_buffer = st.camera_input("Take a photo to analyze")
        if img_file_buffer:
            image = Image.open(img_file_buffer)
            image_np = np.array(image.convert('RGB'))
            
            with st.spinner('Analyzing...'):
                annotated_image, faces = detect_emotion(image_np)
            
            st.image(annotated_image, channels="RGB")

            if faces:
                for i, face in enumerate(faces):
                    st.write(f"**Dominant Emotion: {face['dominant_emotion']}**")
