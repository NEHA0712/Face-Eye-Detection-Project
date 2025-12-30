import cv2
import dlib
from scipy.spatial import distance
import time

# -------------------------------
# Eye Aspect Ratio Function
# -------------------------------
def eye_aspect_ratio(eye):
    # compute euclidean distances
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# -------------------------------
# Constants
# -------------------------------
EYE_AR_THRESH = 0.25   # Eye aspect ratio threshold
EYE_AR_CONSEC_FRAMES = 3  # Consecutive frames for blink

# -------------------------------
# Initialize
# -------------------------------
COUNTER = 0
TOTAL = 0
SLEEP_ALERT = False

# Load face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download from dlib model zoo

(lStart, lEnd) = (42, 48)  # Left eye
(rStart, rEnd) = (36, 42)  # Right eye

# Start camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray, 0)

    for face in faces:
        shape = predictor(gray, face)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        # Extract eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Convert eyes to numpy array for drawing
        leftEyeHull = cv2.convexHull(cv2.array(leftEye))
        rightEyeHull = cv2.convexHull(cv2.array(rightEye))

        # Draw face rectangle
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw eyes convex hull
        cv2.polylines(frame, [leftEyeHull], True, (0, 255, 255), 1)
        cv2.polylines(frame, [rightEyeHull], True, (0, 255, 255), 1)

        # Blink detection
        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            COUNTER = 0

        # Sleep detection (if eyes closed > 2 sec)
        if ear < EYE_AR_THRESH:
            if not SLEEP_ALERT:
                start_sleep = time.time()
                SLEEP_ALERT = True
            elif time.time() - start_sleep >= 2:
                cv2.putText(frame, "SLEEP ALERT!", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        else:
            SLEEP_ALERT = False

        # Display EAR, Blink Count
        cv2.putText(frame, f"Blinks: {TOTAL}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (300,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Show frame
    cv2.imshow("Face + Eye Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
