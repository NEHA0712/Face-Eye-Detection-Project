import cv2
import streamlit as st

# Page config
st.set_page_config(page_title="Eye Blink Detection", layout="centered")

st.title("👁️ Eye Blink Detection App")

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# Variables
blink_counter = 0
eyes_closed_frames = 0

# Start / Stop buttons
start = st.button("Start Camera")
stop = st.button("Stop Camera")

frame_placeholder = st.empty()

if start:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        if stop:
            break

        ret, frame = cap.read()
        if not ret:
            st.warning("Camera not working")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

            if len(eyes) == 0:
                eyes_status = "Closed"
                eyes_closed_frames += 1
            else:
                eyes_status = "Open"
                if eyes_closed_frames >= 2:
                    blink_counter += 1
                eyes_closed_frames = 0

            cv2.putText(frame, f"Eyes: {eyes_status}", (x, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.putText(frame, f"Blink Count: {blink_counter}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")

    cap.release()
