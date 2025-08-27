import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import easyocr
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from datetime import datetime
import tempfile

# -------- Helper Functions ----------
def smooth_line(pts):
    if len(pts) < 3:
        return pts
    smoothed = []
    for i in range(1, len(pts) - 1):
        x = int((pts[i - 1][0] + pts[i][0] + pts[i + 1][0]) / 3)
        y = int((pts[i - 1][1] + pts[i][1] + pts[i + 1][1]) / 3)
        smoothed.append((x, y))
    return smoothed

def save_canvases_to_pdf(canvas_list, filename="airwriting.pdf"):
    pdf_path = os.path.join(tempfile.gettempdir(), filename)
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    for i, img in enumerate(canvas_list):
        # Convert to RGB and resize to fit A4
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        scale = min(width / w, height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img_rgb, (new_w, new_h))

        # Convert to reportlab ImageReader
        img_reader = ImageReader(img_resized)
        x = (width - new_w) // 2
        y = (height - new_h) // 2
        c.drawImage(img_reader, x, y, width=new_w, height=new_h)
        c.showPage()

    c.save()
    return pdf_path

# -------- MediaPipe Init ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.3,
                       min_tracking_confidence=0.3)
mp_draw = mp.solutions.drawing_utils
reader = easyocr.Reader(['en'])

# -------- Streamlit UI -------------
st.set_page_config(page_title="Air Writing", layout="wide")
st.title("✍️ Air Writing with Hand Tracking")

# Session State
if "canvas_list" not in st.session_state:
    st.session_state.canvas_list = [255 * np.ones((480, 640, 3), dtype=np.uint8)]
if "current_page" not in st.session_state:
    st.session_state.current_page = 0
if "draw_color" not in st.session_state:
    st.session_state.draw_color = (0, 0, 255)  # Default red
if "is_drawing" not in st.session_state:
    st.session_state.is_drawing = False
if "draw_points" not in st.session_state:
    st.session_state.draw_points = []
if "mode" not in st.session_state:
    st.session_state.mode = "draw"

# Sidebar controls
st.sidebar.header("Controls 🎛️")
if st.sidebar.button("🧹 Clear Canvas"):
    st.session_state.canvas_list[st.session_state.current_page][:] = 255

if st.sidebar.button("➕ Next Page"):
    st.session_state.canvas_list.append(255 * np.ones((480, 640, 3), dtype=np.uint8))
    st.session_state.current_page += 1

if st.sidebar.button("⬅️ Previous Page"):
    if st.session_state.current_page > 0:
        st.session_state.current_page -= 1

color = st.sidebar.radio("✏️ Pen Color", ["Red", "Green", "Blue", "Black"])
if color == "Red":
    st.session_state.draw_color = (0, 0, 255)
elif color == "Green":
    st.session_state.draw_color = (0, 255, 0)
elif color == "Blue":
    st.session_state.draw_color = (255, 0, 0)
else:
    st.session_state.draw_color = (0, 0, 0)

mode = st.sidebar.radio("🛠 Mode", ["draw", "circle", "rectangle", "text"])
st.session_state.mode = mode

if st.sidebar.button("💾 Download PDF"):
    pdf_file = save_canvases_to_pdf(st.session_state.canvas_list,
                                    filename=f"airwriting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    with open(pdf_file, "rb") as f:
        st.sidebar.download_button("⬇️ Download Canvas PDF", f, file_name="airwriting.pdf")

# Camera Start/Stop
run = st.checkbox("📷 Start Camera")
frame_window = st.image([])

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Camera not working...")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark
        ix, iy = int(lm[8].x * w), int(lm[8].y * h)  # Index finger tip
        tx, ty = int(lm[4].x * w), int(lm[4].y * h)  # Thumb tip
        dist = np.hypot(ix - tx, iy - ty)

        if dist < 40:
            st.session_state.is_drawing = True
            st.session_state.draw_points.append((ix, iy))
        else:
            if st.session_state.is_drawing and st.session_state.draw_points:
                pts = st.session_state.draw_points
                if st.session_state.mode == "circle":
                    (x1, y1), (x2, y2) = pts[0], pts[-1]
                    radius = int(np.hypot(x2 - x1, y2 - y1))
                    cv2.circle(st.session_state.canvas_list[st.session_state.current_page],
                               (x1, y1), radius, st.session_state.draw_color, 3)
                elif st.session_state.mode == "rectangle":
                    (x1, y1), (x2, y2) = pts[0], pts[-1]
                    cv2.rectangle(st.session_state.canvas_list[st.session_state.current_page],
                                  (x1, y1), (x2, y2), st.session_state.draw_color, 3)
                elif st.session_state.mode == "text":
                    x_min = min([p[0] for p in pts])
                    y_min = min([p[1] for p in pts])
                    x_max = max([p[0] for p in pts])
                    y_max = max([p[1] for p in pts])
                    roi = frame[y_min:y_max, x_min:x_max]
                    if roi.size > 0:
                        result = reader.readtext(roi)
                        if result:
                            text = result[0][1]
                            cv2.putText(st.session_state.canvas_list[st.session_state.current_page],
                                        text, (x_min, y_min - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, st.session_state.draw_color, 2)
            st.session_state.is_drawing = False
            st.session_state.draw_points.clear()

        mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    if st.session_state.mode == "draw":
        smoothed_points = smooth_line(list(st.session_state.draw_points))
        for i in range(1, len(smoothed_points)):
            cv2.line(st.session_state.canvas_list[st.session_state.current_page],
                     smoothed_points[i - 1], smoothed_points[i],
                     st.session_state.draw_color, 3)

    display = frame.copy()
    mask = st.session_state.canvas_list[st.session_state.current_page] < 255
    display[mask] = st.session_state.canvas_list[st.session_state.current_page][mask]

    cv2.putText(display, f'Mode: {st.session_state.mode}',
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, st.session_state.draw_color, 2)
    cv2.putText(display, f'Page: {st.session_state.current_page + 1}/{len(st.session_state.canvas_list)}',
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)

    frame_window.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))

cap.release()
