# Face & Eye Detection Project

This project detects faces and eyes using your webcam. It also counts blinks and can help with basic sleep detection. Built using Python, OpenCV, and dlib.

---

## Features

1. **Face Detection**  
   Detect faces in real-time using your webcam. Faces are highlighted with green boxes.

2. **Eye Detection & Blink Count**  
   Detect eyes inside the face region and count blinks in real-time.

3. **Sleep Detection**  
   If eyes remain closed for a certain duration, it alerts as a possible sleep event.

4. **EEG Analysis (Optional)**  
   You can load your EEG model and predict states using uploaded EEG data files.

---

## Requirements

- Python 3.x  
- OpenCV  
- dlib (optional, for advanced eye detection)  
- numpy  
- imutils  
- scipy  

Install required packages using:

```bash
pip install opencv-python numpy imutils scipy dlib
