import cv2
import numpy as np
import pickle
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load models
with open("best_eeg_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

blink_count = 0
closed_frames = 0


def preprocess_eye(eye_img):
    eye = cv2.resize(eye_img, (24, 24))
    eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    eye = eye.flatten().reshape(1, -1)
    eye = scaler.transform(eye)
    return eye


def gen_frames():
    global blink_count, closed_frames
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        eye_status = "Unknown"

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

            if len(eyes) > 0:
                for (ex, ey, ew, eh) in eyes[:1]:
                    eye_img = roi_color[ey:ey+eh, ex:ex+ew]

                    eye_input = preprocess_eye(eye_img)
                    prediction = model.predict(eye_input)[0]

                    if prediction == 1:
                        eye_status = "OPEN"
                        closed_frames = 0
                    else:
                        eye_status = "CLOSED"
                        closed_frames += 1

                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255,0,0), 2)
            else:
                closed_frames += 1
                eye_status = "CLOSED"

            if closed_frames >= 5:
                blink_count += 1
                closed_frames = 0

            cv2.putText(frame, f"Eyes: {eye_status}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.putText(frame, f"Blinks: {blink_count}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
