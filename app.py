import cv2
import mediapipe as mp
import numpy as np
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"

# ── MediaPipe setup ──────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
hands       = mp_hands.Hands(
                  max_num_hands=1,
                  min_detection_confidence=0.7,
                  min_tracking_confidence=0.7)

# ── Webcam ───────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ret, frame = cap.read()
if not ret:
    print("Cannot open webcam!")
    exit()

H, W = frame.shape[:2]

# ── Canvas ───────────────────────────────────────────────────
canvas      = np.zeros((H, W, 3), dtype=np.uint8)
prev_x, prev_y = None, None

# ── Colors palette ───────────────────────────────────────────
COLORS = {
    "Red":    (0,   0,   255),
    "Green":  (0,   255, 0  ),
    "Blue":   (255, 0,   0  ),
    "Yellow": (0,   255, 255),
    "White":  (255, 255, 255),
    "Eraser": (0,   0,   0  ),
}
COLOR_NAMES  = list(COLORS.keys())
current_color = COLORS["Red"]
brush_size    = 8
PALETTE_Y     = 20
PALETTE_H     = 60
BOX_W         = W // len(COLOR_NAMES)

# ── Finger detection ─────────────────────────────────────────
def fingers_up(hand_landmarks):
    tips    = [8, 12, 16, 20]   # index, middle, ring, pinky
    base    = [6, 10, 14, 18]
    fingers = []
    # Thumb
    fingers.append(
        hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x)
    # Other fingers
    for t, b in zip(tips, base):
        fingers.append(
            hand_landmarks.landmark[t].y < hand_landmarks.landmark[b].y)
    return fingers   # [thumb, index, middle, ring, pinky]

def draw_palette(frame):
    for i, (name, color) in enumerate(COLORS.items()):
        x1 = i * BOX_W
        x2 = x1 + BOX_W
        cv2.rectangle(frame, (x1, PALETTE_Y),
                      (x2, PALETTE_Y + PALETTE_H), color, -1)
        # Highlight selected color
        if color == current_color:
            cv2.rectangle(frame, (x1, PALETTE_Y),
                          (x2, PALETTE_Y + PALETTE_H), (255,255,255), 3)
        cv2.putText(frame, name, (x1 + 5, PALETTE_Y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

def get_palette_color(cx, cy):
    if PALETTE_Y <= cy <= PALETTE_Y + PALETTE_H:
        idx = cx // BOX_W
        if 0 <= idx < len(COLOR_NAMES):
            return COLORS[COLOR_NAMES[idx]]
    return None

def draw_ui(frame, mode, finger_count):
    # Mode label
    mode_color = (0,255,0) if mode == "DRAWING" else \
                 (0,165,255) if mode == "PAUSE" else (0,0,255)
    cv2.rectangle(frame, (0, H-50), (300, H), (0,0,0), -1)
    cv2.putText(frame, f"Mode: {mode}", (10, H-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
    # Finger count
    cv2.putText(frame, f"Fingers: {finger_count}", (320, H-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    # Instructions
    cv2.putText(frame, "1 finger=Draw | 5 fingers=Pause | Fist=Clear | S=Save",
                (10, H - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

# ── Main loop ────────────────────────────────────────────────
print("Air Drawing App started!")
print("Controls:")
print("  Index finger only → Draw")
print("  All 5 fingers up  → Pause")
print("  Fist (0 fingers)  → Clear canvas")
print("  Hover palette     → Change color")
print("  S key             → Save drawing")
print("  Q key             → Quit")

mode = "PAUSE"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand
    result = hands.process(rgb)

    finger_count = 0

    if result.multi_hand_landmarks:
        for hand_lm in result.multi_hand_landmarks:
            # Draw hand skeleton
            mp_drawing.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            # Get finger states
            f = fingers_up(hand_lm)
            finger_count = sum(f)

            # Get index fingertip position
            lm = hand_lm.landmark
            ix = int(lm[8].x * W)
            iy = int(lm[8].y * H)

            # ── Gesture detection ────────────────────────────
            only_index = f[1] and not f[2] and not f[3] and not f[4]
            all_up     = all(f[1:])   # index+middle+ring+pinky up
            fist       = finger_count == 0

            # Check palette hover
            palette_hit = get_palette_color(ix, iy)

            if fist:
                # CLEAR
                canvas    = np.zeros((H, W, 3), dtype=np.uint8)
                prev_x, prev_y = None, None
                mode      = "CLEAR"

            elif all_up:
                # PAUSE
                prev_x, prev_y = None, None
                mode      = "PAUSE"

                # Color selection when hovering palette
                if palette_hit is not None:
                    current_color = palette_hit

            elif only_index:
                # DRAW
                mode = "DRAWING"

                if palette_hit is not None:
                    # Hovering palette — change color, don't draw
                    current_color  = palette_hit
                    prev_x, prev_y = None, None
                else:
                    if prev_x is not None and prev_y is not None:
                        color = (0,0,0) if current_color == COLORS["Eraser"] \
                                else current_color
                        cv2.line(canvas,
                                 (prev_x, prev_y), (ix, iy),
                                 color, brush_size)
                    prev_x, prev_y = ix, iy

            else:
                prev_x, prev_y = None, None
                mode = "PAUSE"

            # Draw cursor circle
            cv2.circle(frame, (ix, iy), 10, current_color, -1)
            cv2.circle(frame, (ix, iy), 12, (255,255,255), 2)

    else:
        prev_x, prev_y = None, None
        mode = "NO HAND"

    # ── Merge canvas with frame ──────────────────────────────
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask     = cv2.threshold(canvas_gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv    = cv2.bitwise_not(mask)
    frame_bg    = cv2.bitwise_and(frame, frame,  mask=mask_inv)
    canvas_fg   = cv2.bitwise_and(canvas, canvas, mask=mask)
    combined    = cv2.add(frame_bg, canvas_fg)

    # ── Draw UI ──────────────────────────────────────────────
    draw_palette(combined)
    draw_ui(combined, mode, finger_count)

    cv2.imshow("Air Drawing App", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = "drawing.png"
        cv2.imwrite(filename, canvas)
        print(f"Drawing saved as {filename}")
        cv2.putText(combined, "SAVED!", (W//2 - 60, H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)

cap.release()
cv2.destroyAllWindows()
print("App closed.")
