import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
import pytesseract
from collections import deque

# --- Smooth line helper ---
def smooth_line(pts):
    if len(pts) < 3:
        return pts
    smoothed = []
    for i in range(1, len(pts) - 1):
        x = int((pts[i - 1][0] + pts[i][0] + pts[i + 1][0]) / 3)
        y = int((pts[i - 1][1] + pts[i][1] + pts[i + 1][1]) / 3)
        smoothed.append((x, y))
    return smoothed

# --- Initialize MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# --- Global state ---
if "colors" not in st.session_state:
    st.session_state.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
if "color_index" not in st.session_state:
    st.session_state.color_index = 0
if "draw_color" not in st.session_state:
    st.session_state.draw_color = st.session_state.colors[0]
if "mode" not in st.session_state:
    st.session_state.mode = "draw"
if "draw_points" not in st.session_state:
    st.session_state.draw_points = deque(maxlen=512)
if "canvas_list" not in st.session_state:
    st.session_state.canvas_list = []
if "current_page" not in st.session_state:
    st.session_state.current_page = 0

# --- Button callbacks ---
def change_color():
    st.session_state.color_index = (st.session_state.color_index + 1) % len(st.session_state.colors)
    st.session_state.draw_color = st.session_state.colors[st.session_state.color_index]
    st.session_state.mode = "draw"

def set_erase():
    st.session_state.draw_color = (255, 255, 255)
    st.session_state.mode = "erase"

def set_draw():
    st.session_state.draw_color = st.session_state.colors[st.session_state.color_index]
    st.session_state.mode = "draw"

def new_page(h, w):
    st.session_state.canvas_list.append(255 * np.ones((h, w, 3), dtype=np.uint8))
    st.session_state.current_page = len(st.session_state.canvas_list) - 1

def prev_page():
    if st.session_state.current_page > 0:
        st.session_state.current_page -= 1

def next_page():
    if st.session_state.current_page < len(st.session_state.canvas_list) - 1:
        st.session_state.current_page += 1

def save_canvas():
    cv2.imwrite("canvas_output.png", st.session_state.canvas_list[st.session_state.current_page])
    st.sidebar.success("✅ Canvas saved as canvas_output.png")

def recognize_text():
    img = st.session_state.canvas_list[st.session_state.current_page]
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    text = pytesseract.image_to_string(pil_img)
    st.sidebar.subheader("📖 Recognized Text:")
    st.sidebar.write(text if text.strip() else "No text detected.")

# --- Streamlit UI ---
st.title("✍️ Air Writing with Hand Tracking + Controls")
run = st.checkbox("Start Camera")

# --- Sidebar buttons ---
if st.sidebar.button("Change color"): change_color()
if st.sidebar.button("Erase Mode"): set_erase()
if st.sidebar.button("Draw Mode"): set_draw()
if st.sidebar.button("Previous Page"): prev_page()
if st.sidebar.button("Next Page"): next_page()
if st.sidebar.button("Recognize text"): recognize_text()
if st.sidebar.button("Save Canvas"): save_canvas()

# --- Camera frame ---
frame_window = st.image([])

if run:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)   # 🔽 lower resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 140)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        st.write("❌ Camera not working...")
    else:
        h, w, _ = frame.shape

        # If no canvas yet, create the first one matching camera resolution
        if len(st.session_state.canvas_list) == 0:
            new_page(h, w)

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Hand detection
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            ix, iy = int(lm[8].x * w), int(lm[8].y * h)  # Index fingertip
            tx, ty = int(lm[4].x * w), int(lm[4].y * h)  # Thumb tip
            dist = np.hypot(ix - tx, iy - ty)

            if dist < 40:  # Pinch gesture
                if st.session_state.mode == "draw":
                    st.session_state.draw_points.append((ix, iy))
                elif st.session_state.mode == "erase":
                    cv2.circle(st.session_state.canvas_list[st.session_state.current_page], (ix, iy), 20, (255, 255, 255), -1)
            else:
                st.session_state.draw_points.clear()

            mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        # Drawing
        if st.session_state.mode == "draw" and len(st.session_state.draw_points) > 1:
            smoothed_points = smooth_line(list(st.session_state.draw_points))
            for i in range(1, len(smoothed_points)):
                cv2.line(
                    st.session_state.canvas_list[st.session_state.current_page],
                    smoothed_points[i - 1],
                    smoothed_points[i],
                    st.session_state.draw_color,
                    3
                )

        # Overlay current canvas on live video
        display = frame.copy()
        mask = st.session_state.canvas_list[st.session_state.current_page] < 255
        display[mask] = st.session_state.canvas_list[st.session_state.current_page][mask]

        cv2.putText(display, f'Mode: {st.session_state.mode}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, st.session_state.draw_color, 2)
        cv2.putText(display, f'Page: {st.session_state.current_page + 1}/{len(st.session_state.canvas_list)}',
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)

        frame_window.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
