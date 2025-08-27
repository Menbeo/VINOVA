import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from typing import List, Tuple
from PIL import Image
import io
import threading

# ---------- Helper ----------
def smooth_line(pts: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if len(pts) < 3:
        return pts
    smoothed = []
    for i in range(1, len(pts) - 1):
        x = int((pts[i - 1][0] + pts[i][0] + pts[i + 1][0]) / 3)
        y = int((pts[i - 1][1] + pts[i][1] + pts[i + 1][1]) / 3)
        smoothed.append((x, y))
    return smoothed

# ---------- Transformer ----------
class HandDrawer(VideoTransformerBase):
    def __init__(self):
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # drawing state (kept inside transformer for low-latency)
        self.canvas_list: List[np.ndarray] = [255 * np.ones((480, 640, 3), dtype=np.uint8)]
        self.current_page: int = 0
        self.draw_color: Tuple[int, int, int] = (0, 0, 255)  # BGR
        self.draw_points: List[Tuple[int, int]] = []
        self.is_drawing: bool = False

        # lock to safely mutate canvas from main thread
        self.lock = threading.Lock()

    def transform(self, frame):
        # frame is av.VideoFrame-like; convert to ndarray BGR
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        # Resize canvas to match incoming frame size if needed
        with self.lock:
            if self.canvas_list[self.current_page].shape[0] != h or self.canvas_list[self.current_page].shape[1] != w:
                # recreate canvases with the correct size
                new_canvas = 255 * np.ones((h, w, 3), dtype=np.uint8)
                self.canvas_list = [new_canvas.copy() for _ in range(len(self.canvas_list))]

        # Process with MediaPipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            ix, iy = int(lm[8].x * w), int(lm[8].y * h)  # index finger tip
            tx, ty = int(lm[4].x * w), int(lm[4].y * h)  # thumb tip
            dist = np.hypot(ix - tx, iy - ty)

            if dist < 40:
                self.is_drawing = True
                self.draw_points.append((ix, iy))
            else:
                self.is_drawing = False
                # keep last stroke on canvas and clear points
                self.draw_points.clear()

            # draw hand landmarks small for feedback
            self.mp_draw.draw_landmarks(img, results.multi_hand_landmarks[0], self.mp_hands.HAND_CONNECTIONS)

        # Draw smoothed lines onto canvas
        smoothed_points = smooth_line(list(self.draw_points))
        with self.lock:
            canvas = self.canvas_list[self.current_page]
            for i in range(1, len(smoothed_points)):
                cv2.line(canvas,
                         smoothed_points[i - 1], smoothed_points[i],
                         self.draw_color, 5, lineType=cv2.LINE_AA)

            # overlay canvas on the frame: where canvas is not white (<255) copy
            mask = np.any(canvas < 255, axis=2)
            img[mask] = canvas[mask]

        # add HUD text
        cv2.putText(img, f'Page: {self.current_page + 1}/{len(self.canvas_list)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
        return img

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Air Writing (webrtc)", layout="wide")
st.title("✍️ Air Writing (Realtime) — Streamlit + streamlit-webrtc + MediaPipe")

st.sidebar.header("Controls")

# color selection (BGR values)
color_name = st.sidebar.selectbox("Pen color", ["Red", "Green", "Blue", "Black"])
color_map = {"Red": (0, 0, 255), "Green": (0, 255, 0), "Blue": (255, 0, 0), "Black": (0, 0, 0)}
selected_color = color_map[color_name]

# Create/Start webrtc streamer
webrtc_ctx = webrtc_streamer(
    key="hand-drawer",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=HandDrawer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.sidebar.markdown("---")

col1, col2, col3 = st.sidebar.columns([1,1,1])
with col1:
    if st.button("➕ Next Page"):
        if webrtc_ctx.state.playing and webrtc_ctx.video_transformer:
            with webrtc_ctx.video_transformer.lock:
                # append new white canvas same size as current
                h, w = webrtc_ctx.video_transformer.canvas_list[0].shape[:2]
                webrtc_ctx.video_transformer.canvas_list.append(255 * np.ones((h, w, 3), dtype=np.uint8))
                webrtc_ctx.video_transformer.current_page = len(webrtc_ctx.video_transformer.canvas_list) - 1

with col2:
    if st.button("⬅️ Prev Page"):
        if webrtc_ctx.state.playing and webrtc_ctx.video_transformer:
            with webrtc_ctx.video_transformer.lock:
                if webrtc_ctx.video_transformer.current_page > 0:
                    webrtc_ctx.video_transformer.current_page -= 1

with col3:
    if st.button("🧹 Clear Page"):
        if webrtc_ctx.state.playing and webrtc_ctx.video_transformer:
            with webrtc_ctx.video_transformer.lock:
                idx = webrtc_ctx.video_transformer.current_page
                h, w = webrtc_ctx.video_transformer.canvas_list[idx].shape[:2]
                webrtc_ctx.video_transformer.canvas_list[idx][:] = 255

st.sidebar.markdown("---")

if st.sidebar.button("Save Current Page as PNG"):
    if webrtc_ctx.state.playing and webrtc_ctx.video_transformer:
        with webrtc_ctx.video_transformer.lock:
            img = webrtc_ctx.video_transformer.canvas_list[webrtc_ctx.video_transformer.current_page]
            # convert to PNG bytes
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            buf.seek(0)
            st.sidebar.download_button("Download PNG", data=buf, file_name=f"page_{webrtc_ctx.video_transformer.current_page+1}.png", mime="image/png")

st.sidebar.markdown("---")
st.sidebar.write("Notes:")
st.sidebar.write("- Allow camera permission when prompted.")
st.sidebar.write("- If the app is laggy, try reducing resolution or switching browser.")

# Update transformer's color in real-time
if webrtc_ctx.state.playing and webrtc_ctx.video_transformer:
    with webrtc_ctx.video_transformer.lock:
        webrtc_ctx.video_transformer.draw_color = selected_color

# Show a small preview of current canvas and info
st.sidebar.markdown("### Current Canvas Preview")
if webrtc_ctx.state.playing and webrtc_ctx.video_transformer:
    with webrtc_ctx.video_transformer.lock:
        tmp = webrtc_ctx.video_transformer.canvas_list[webrtc_ctx.video_transformer.current_page]
        # convert to RGB for PIL/streamlit
        tmp_rgb = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
        st.sidebar.image(tmp_rgb, use_column_width=True)
        st.sidebar.write(f"Page {webrtc_ctx.video_transformer.current_page + 1} of {len(webrtc_ctx.video_transformer.canvas_list)}")
else:
    st.sidebar.write("WebRTC not started yet or no transformer instance.")
