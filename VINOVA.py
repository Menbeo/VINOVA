import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

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

# -------- MediaPipe Init ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.3)
mp_draw = mp.solutions.drawing_utils

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

# Camera Start/Stop
run = st.checkbox("📷 Start Camera")
frame_window = st.image([])

cap = cv2.VideoCapture(0)

# ✅ Lower resolution to reduce lag
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Try 320 for more speed
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Try 240 for more speed

frame_skip = 2  # Process every 2nd frame (set 1 = process all frames)
frame_count = 0

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Camera not working...")
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1

    # Skip some frames to improve performance
    if frame_count % frame_skip != 0:
        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        continue

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

    frame_window.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))

cap.release()
