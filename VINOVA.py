import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import easyocr
import math
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
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

def recognize_text(img):
    reader = easyocr.Reader(["en"], gpu=False)
    results = reader.readtext(img)
    return " ".join([res[1] for res in results])

def recognize_shape(points):
    if len(points) < 5:
        return None
    cnt = np.array(points).reshape((-1, 1, 2)).astype(np.int32)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    if len(approx) == 3:
        return "Triangle"
    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        return "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
    elif len(approx) > 4:
        return "Circle"
    return None

def export_to_pdf(canvas_list):
    """Save all pages to a multi-page PDF and return the file path"""
    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    c = canvas.Canvas(pdf_path, pagesize=A4)
    page_w, page_h = A4

    for img in canvas_list:
        # Convert numpy canvas to ImageReader
        _, buf = cv2.imencode(".png", img)
        image = ImageReader(buf.tobytes())

        # Scale to fit A4
        iw, ih = img.shape[1], img.shape[0]
        scale = min(page_w / iw, page_h / ih)
        new_w, new_h = iw * scale, ih * scale

        # Center on page
        x = (page_w - new_w) / 2
        y = (page_h - new_h) / 2

        c.drawImage(image, x, y, width=new_w, height=new_h)
        c.showPage()

    c.save()
    return pdf_path

# -------- MediaPipe Init ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.3,
                       min_tracking_confidence=0.3)
mp_draw = mp.solutions.drawing_utils

# -------- Streamlit UI -------------
st.set_page_config(page_title="Air Writing", layout="wide")
st.title("✍️ Air Writing with Hand Tracking + Text, Shape & PDF Export")

# Session State
if "canvas_list" not in st.session_state:
    st.session_state.canvas_list = [255 * np.ones((480, 640, 3), dtype=np.uint8)]
if "current_page" not in st.session_state:
    st.session_state.current_page = 0
if "draw_color" not in st.session_state:
    st.session_state.draw_color = (0, 0, 255)
if "is_drawing" not in st.session_state:
    st.session_state.is_drawing = False
if "draw_points" not in st.session_state:
    st.session_state.draw_points = []
if "mode" not in st.session_state:
    st.session_state.mode = "draw"
if "last_recognized_text" not in st.session_state:
    st.session_state.last_recognized_text = ""
if "last_recognized_shape" not in st.session_state:
    st.session_state.last_recognized_shape = ""

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

# Recognition buttons
if st.sidebar.button("🔤 Recognize Text"):
    canvas_img = st.session_state.canvas_list[st.session_state.current_page]
    gray = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
    st.session_state.last_recognized_text = recognize_text(gray)

if st.sidebar.button("🔺 Recognize Shape"):
    shape = recognize_shape(st.session_state.draw_points)
    if shape:
        st.session_state.last_recognized_shape = shape

# Export PDF
if st.sidebar.button("📥 Download as PDF"):
    pdf_file = export_to_pdf(st.session_state.canvas_list)
    with open(pdf_file, "rb") as f:
        st.download_button("⬇️ Click to Save PDF", f, file_name="air_writing.pdf", mime="application/pdf")

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
            st.session_state.is_drawing = False
            st.session_state.draw_points.clear()

        mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

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

    if st.session_state.last_recognized_text:
        cv2.putText(display, f'Text: {st.session_state.last_recognized_text}',
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 2)

    if st.session_state.last_recognized_shape:
        cv2.putText(display, f'Shape: {st.session_state.last_recognized_shape}',
                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 0), 2)

    frame_window.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))

cap.release()
