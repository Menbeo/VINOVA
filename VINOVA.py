
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

# Smooth line helper
def smooth_line(pts):
    if len(pts) < 3:
        return pts
    smoothed = []
    for i in range(1, len(pts) - 1):
        x = int((pts[i - 1][0] + pts[i][0] + pts[i + 1][0]) / 3)
        y = int((pts[i - 1][1] + pts[i][1] + pts[i + 1][1]) / 3)
        smoothed.append((x, y))
    return smoothed

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Global variables
is_drawing = False
draw_points = []
draw_color = (0, 0, 255)  # Red
canvas_list = [255 * np.ones((480, 640, 3), dtype=np.uint8)]  # One white canvas
current_page = 0
mode = "draw"

# Streamlit UI
st.title("✍️ Air Writing with Hand Tracking")
run = st.checkbox("Start Camera")

frame_window = st.image([])  # Placeholder for video frames

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Camera not working...")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark
        ix, iy = int(lm[8].x * w), int(lm[8].y * h)
        tx, ty = int(lm[4].x * w), int(lm[4].y * h)
        dist = np.hypot(ix - tx, iy - ty)

        if dist < 40:
            is_drawing = True
            draw_points.append((ix, iy))
        else:
            is_drawing = False
            draw_points.clear()

        mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    smoothed_points = smooth_line(list(draw_points))
    for i in range(1, len(smoothed_points)):
        cv2.line(canvas_list[current_page], smoothed_points[i - 1], smoothed_points[i], draw_color, 3)

    display = frame.copy()
    mask = canvas_list[current_page] < 255
    display[mask] = canvas_list[current_page][mask]

    cv2.putText(display, f'Mode: {mode}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)
    cv2.putText(display, f'Page: {current_page + 1}/{len(canvas_list)}', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)

    frame_window.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))

cap.release()