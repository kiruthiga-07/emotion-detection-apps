import streamlit as st
import cv2
from fer import FER
import numpy as np
from PIL import Image

# ----------------- Page Setup -----------------
st.set_page_config(page_title="Emotion Detection App", layout="centered")

# ----------------- Welcome Page -----------------
st.title("🎭 Welcome to the Emotion Detection App!")
st.write("""
This app allows you to:
- Detect emotions from uploaded images
- Detect emotions in real-time using your webcam
""")
st.write("Click below to continue to the app:")

if st.button("Go to Emotion Detection"):
    st.title("🎭 Real-Time Emotion Detection App")

    # ----------------- Sidebar Options -----------------
    option = st.sidebar.selectbox(
        "Select Input Mode",
        ("Upload Image", "Use Webcam")
    )

    # Initialize FER detector
    detector = FER(mtcnn=True)

    # ----------------- Function to detect emotion -----------------
    def detect_emotion(image):
        result = detector.detect_emotions(image)
        for face in result:
            (x, y, w, h) = face["box"]
            emotions = face["emotions"]
            dominant_emotion = max(emotions, key=emotions.get)
            # Draw rectangle around face
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Write dominant emotion
            cv2.putText(image, dominant_emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        return image

    # ----------------- Upload Image Mode -----------------
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            image_np = np.array(image.convert('RGB'))
            annotated_image = detect_emotion(image_np)
            st.image(annotated_image, channels="RGB", caption="Emotion Detection")

    # ----------------- Webcam Mode -----------------
    elif option == "Use Webcam":
        st.info("Click 'Start Webcam' to detect emotions in real-time")
        run = st.button("Start Webcam")
        FRAME_WINDOW = st.image([])

        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Unable to access webcam")
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_frame = detect_emotion(frame)
            FRAME_WINDOW.image(annotated_frame)

        cap.release()
