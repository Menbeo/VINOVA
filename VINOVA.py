import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

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


# --- Globals ---
draw_color = (0, 0, 255)   # red
is_drawing = False
draw_points = []
canvas_list = [255 * np.ones((480, 640, 3), np.uint8)]  # one white canvas
current_page = 0
mode = "draw"

# --- Mediapipe hands ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils


# --- Streamlit Video Transformer ---
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.canvas_list = canvas_list
        self.current_page = current_page
        self.draw_color = draw_color
        self.draw_points = []
        self.is_drawing = False

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        h, w, _ = img.shape

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            ix, iy = int(lm[8].x * w), int(lm[8].y * h)   # index fingertip
            tx, ty = int(lm[4].x * w), int(lm[4].y * h)   # thumb tip
            dist = np.hypot(ix - tx, iy - ty)

            if dist < 40:
                self.is_drawing = True
                self.draw_points.append((ix, iy))
            else:
                self.is_drawing = False
                self.draw_points.clear()

            mp_draw.draw_landmarks(img, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        # Draw smoothed line on canvas
        smoothed_points = smooth_line(list(self.draw_points))
        for i in range(1, len(smoothed_points)):
            cv2.line(self.canvas_list[self.current_page], smoothed_points[i - 1], smoothed_points[i], self.draw_color, 3)

        # Overlay canvas on video
        display = img.copy()
        mask = self.canvas_list[self.current_page] < 255
        display[mask] = self.canvas_list[self.current_page][mask]

        cv2.putText(display, f'Mode: {mode}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.draw_color, 2)
        cv2.putText(display, f'Page: {self.current_page+1}/{len(self.canvas_list)}',
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)

        return display


# --- Streamlit UI ---
st.title("✍️ Air Writing with MediaPipe + Streamlit")

webrtc_streamer(
    key="air-writing",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
)
