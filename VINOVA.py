import cv2
import numpy as np
import streamlit as st
from collections import deque
import pytesseract

# ------------------------------
# Streamlit App Title
# ------------------------------
st.title("✍️ VINOVA - Gesture Drawing with Shape & Text Recognition")

# ------------------------------
# Global session state
# ------------------------------
if "draw_points" not in st.session_state:
    st.session_state.draw_points = deque(maxlen=512)
if "draw_color" not in st.session_state:
    st.session_state.draw_color = (0, 0, 255)  # red default
if "canvas_list" not in st.session_state:
    st.session_state.canvas_list = []
if "current_page" not in st.session_state:
    st.session_state.current_page = 0
if "mode" not in st.session_state:
    st.session_state.mode = "draw"
if "saved_canvases" not in st.session_state:
    st.session_state.saved_canvases = []

# ------------------------------
# Helper Functions
# ------------------------------
def set_draw():
    st.session_state.mode = "draw"
    st.sidebar.success("✏️ Draw Mode Activated")

def set_erase():
    st.session_state.mode = "erase"
    st.sidebar.success("🧹 Erase Mode Activated")

def prev_page():
    if st.session_state.current_page > 0:
        st.session_state.current_page -= 1

def next_page():
    st.session_state.current_page += 1
    if len(st.session_state.canvas_list) <= st.session_state.current_page:
        st.session_state.canvas_list.append(255*np.ones((240,320,3), dtype=np.uint8))

def recognize_text(canvas):
    # Convert to grayscale and run Tesseract OCR
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip()

def draw_shapes(canvas):
    # Convert to gray
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04*cv2.arcLength(cnt, True), True)

        if len(approx) == 3:  # Triangle
            cv2.drawContours(canvas, [approx], 0, (0,255,0), 2)
        elif len(approx) == 4:  # Rectangle or Square
            cv2.drawContours(canvas, [approx], 0, (255,0,0), 2)
        elif len(approx) > 4:  # Circle-like
            (x,y),radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(canvas, center, radius, (0,0,255), 2)
    return canvas

def save_canvas():
    canvas = st.session_state.canvas_list[st.session_state.current_page].copy()

    # Ensure white background
    h, w, _ = canvas.shape
    white_bg = 255 * np.ones((h, w, 3), dtype=np.uint8)
    mask = np.any(canvas < 255, axis=-1)
    white_bg[mask] = canvas[mask]

    filename = f"canvas_output_{len(st.session_state.saved_canvases)+1}.png"
    cv2.imwrite(filename, white_bg)

    # Save in session gallery
    st.session_state.saved_canvases.append(white_bg)

    st.success(f"✅ Canvas saved as {filename}")

# ------------------------------
# Sidebar buttons
# ------------------------------
if st.sidebar.button("Erase Mode"): set_erase()
if st.sidebar.button("Draw Mode"): set_draw()
if st.sidebar.button("Previous Page"): prev_page()
if st.sidebar.button("Next Page"): next_page()
if st.sidebar.button("Shape Draw"):
    st.session_state.canvas_list[st.session_state.current_page] = draw_shapes(
        st.session_state.canvas_list[st.session_state.current_page]
    )
    st.sidebar.success("✅ Shapes refined")
if st.sidebar.button("Recognize text"):
    texts = recognize_text(st.session_state.canvas_list[st.session_state.current_page])
    st.sidebar.write("Detected text:", texts)
if st.sidebar.button("Save Canvas"): save_canvas()

# ------------------------------
# Camera Stream
# ------------------------------
run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)

    # ✅ Set resolution for less lag
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    ret, frame = cap.read()
    cap.release()

    if len(st.session_state.canvas_list) == 0:
        st.session_state.canvas_list.append(255*np.ones((240,320,3), dtype=np.uint8))

    display = st.session_state.canvas_list[st.session_state.current_page].copy()

    # Example: overlay live camera for hand-tracking (placeholder)
    display = cv2.addWeighted(display, 1, frame, 0.2, 0)

    FRAME_WINDOW.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))

# ------------------------------
# Gallery
# ------------------------------
st.subheader("🖼️ Saved Canvases Gallery")
if len(st.session_state.saved_canvases) > 0:
    cols = st.columns(3)
    for idx, img in enumerate(st.session_state.saved_canvases):
        with cols[idx % 3]:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Canvas {idx+1}")
else:
    st.info("No saved canvases yet. Save one to see it here!")
