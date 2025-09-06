
import cv2
import numpy as np
import mediapipe as mp
import easyocr
import os
import tkinter as tk
from threading import Thread
from datetime import datetime
from collections import deque

# Initialize MediaPipe and EasyOCR
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils
reader = easyocr.Reader(['en'])

# Drawing settings
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
color_index = 0
draw_color = colors[color_index]
mode = 'draw'

# Canvas storage
canvas_list = [np.ones((480, 640, 3), dtype=np.uint8) * 255]
current_page = 0
draw_points = deque(maxlen=512)
is_drawing = False

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

# Save current canvas
def save_canvas():
    global canvas_list, current_page
    filename = f"airwriting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(filename, canvas_list[current_page])
    print(f"Saved: {filename}")

# Recognize text from canvas
def recognize_text():
    global canvas_list, current_page
    gray = cv2.cvtColor(canvas_list[current_page], cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray)
    print("Recognized text:")
    for res in results:
        print(res[1])

# UI button callbacks
def change_color():
    global color_index, draw_color, mode
    color_index = (color_index + 1) % len(colors)
    draw_color = colors[color_index]
    mode = 'draw'

def set_erase():
    global draw_color, mode
    draw_color = (255, 255, 255)
    mode = 'erase'

def set_draw():
    global draw_color, mode
    draw_color = colors[color_index]
    mode = 'draw'

def new_page():
    global canvas_list, current_page
    canvas_list.append(np.ones((480, 640, 3), dtype=np.uint8) * 255)
    current_page = len(canvas_list) - 1

def prev_page():
    global current_page
    if current_page > 0:
        current_page -= 1

def next_page():
    global current_page
    if current_page < len(canvas_list) - 1:
        current_page += 1

# Camera and drawing thread
def camera_loop():
    global is_drawing, draw_points, draw_color, canvas_list, current_page, mode
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
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
        cv2.imshow("Air Writing", display)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Build basic Tkinter UI
def build_ui():
    root = tk.Tk()
    root.title("Air Writing Controls")

    tk.Button(root, text="Change Color", command=change_color).pack(fill='x')
    tk.Button(root, text="Erase Mode", command=set_erase).pack(fill='x')
    tk.Button(root, text="Draw Mode", command=set_draw).pack(fill='x')
    tk.Button(root, text="New Page", command=new_page).pack(fill='x')
    tk.Button(root, text="Previous Page", command=prev_page).pack(fill='x')
    tk.Button(root, text="Next Page", command=next_page).pack(fill='x')
    tk.Button(root, text="Save Canvas", command=save_canvas).pack(fill='x')
    tk.Button(root, text="Recognize Text", command=recognize_text).pack(fill='x')
    tk.Button(root, text="Quit", command=root.quit).pack(fill='x')

    root.mainloop()

# Launch both threads
Thread(target=build_ui).start()
camera_loop()