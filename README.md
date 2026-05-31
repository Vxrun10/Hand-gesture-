# ✋ Hand Gesture Analyzer

A real-time hand gesture analyzer developed using Python and OpenCV. The project detects a hand from a webcam feed, estimates the number of visible fingers, and tracks basic hand movements using computer vision techniques.

This project was built to explore contour-based hand detection and understand how gesture recognition can be implemented without using machine learning models.

---

## Features

* Real-time webcam-based hand detection
* Finger counting (0–5 fingers)
* Hand pose recognition

  * Fist
  * One Finger
  * Two Fingers
  * Three Fingers
  * Four Fingers
  * Open Palm
* Motion tracking

  * Moving Left
  * Moving Right
  * Moving Up
  * Moving Down
  * Static
* Adjustable HSV controls for different lighting conditions
* Convex Hull and Convexity Defect visualization
* FPS monitoring

---

## Tech Stack

* Python
* OpenCV
* NumPy

---

## Working

The application captures live video from the webcam and converts each frame into HSV color space. Skin-colored regions are extracted using adjustable HSV thresholds. After noise reduction, the largest contour is selected as the hand region.

Convex Hulls and Convexity Defects are then used to estimate the number of visible fingers. The center of the hand is tracked across frames to determine movement direction.

---

## Project Structure

```text
HandGestureAnalyzer/
│
├── gesture.py
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/HandGestureAnalyzer.git
cd HandGestureAnalyzer
```

Install dependencies:

```bash
pip install opencv-python numpy
```

---

## Run

```bash
python gesture.py
```

Press **ESC** to exit the application.

---

## Challenges Faced

Some of the main challenges during development were:

* Selecting HSV values that work under different lighting conditions
* Reducing background noise
* Improving finger counting accuracy
* Handling different hand orientations
* Maintaining stable motion tracking

---

## Future Improvements

Planned enhancements include:

* MediaPipe-based hand tracking
* Virtual mouse control
* Volume and brightness control using gestures
* Additional gesture support
* Multi-hand detection
* Sign language recognition
* Machine learning-based gesture classification

---

## What I Learned

Through this project, I gained hands-on experience with:

* Image processing using OpenCV
* HSV color segmentation
* Contour analysis
* Convex Hulls and Convexity Defects
* Real-time video processing
* Motion tracking techniques

---

## Author

**Varun Panchal**

B.Tech Student | Computer Vision & AI Enthusiast

---

If you found this project interesting, feel free to fork it or suggest improvements.
