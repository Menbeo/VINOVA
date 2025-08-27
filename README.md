# ✍️ Air Writing with Hand Tracking

This project is a **gesture-based drawing tool** that lets you **write in the air** using only your hand — no pen, tablet, or touchscreen needed.  
It uses **MediaPipe** for hand tracking and **Streamlit** for the interactive web interface.  

---

## 🚀 Features
- Track your hand in real-time using webcam.
- Draw on a virtual canvas by touching **index finger & thumb together**.
- Multi-page support (add next/previous pages).
- Change pen colors (Red, Green, Blue, Black).
- Clear canvas anytime.
- Lower resolution + frame skipping to reduce lag.

---

## 🛠️ Requirements
- Python **3.9 – 3.11** (tested on 3.10)
- Packages:
  ```bash
  pip install opencv-python mediapipe streamlit numpy
