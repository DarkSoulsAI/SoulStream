"""
Hand tracker visual test — opens webcam and draws landmarks +
open-palm detection state in real-time for debugging.

Usage:
    python test_hand_tracker.py

Press Q or ESC to quit.
"""

import cv2
import numpy as np
import time

from hand_tracker import HandTracker, HandData

# MediaPipe hand skeleton connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
]

FINGERTIPS = {4, 8, 12, 16, 20}
FINGER_TIP_TO_NAME = {4: "thumb", 8: "index", 12: "middle", 16: "ring", 20: "pinky"}

# Colors (BGR)
COLOR_OPEN = (0, 255, 0)
COLOR_CLOSED = (0, 0, 255)
COLOR_BONE = (180, 180, 180)
COLOR_JOINT = (255, 255, 0)
COLOR_PALM = (0, 0, 255)
COLOR_TEXT_BG = (30, 30, 30)


def draw_hand_overlay(frame, hand_data, tracker):
    """Draw hand skeleton, finger status, and detection info on the frame."""
    h, w = frame.shape[:2]

    if not hand_data.detected or hand_data.landmarks is None:
        cv2.putText(frame, "No hand detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"EMA confidence: {tracker._ema_confidence:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        return

    lm_ndc = hand_data.landmarks  # list of 21 (ndc_x, ndc_y)

    # Convert NDC back to pixel coords for drawing on the mirrored frame
    # NDC was computed as: ndc_x = 1.0 - x * 2.0, ndc_y = 1.0 - y * 2.0
    # So: x = (1.0 - ndc_x) / 2.0, y = (1.0 - ndc_y) / 2.0
    # But frame is mirrored, so pixel_x = (1 - x) * w = ((1 + ndc_x) / 2) * w
    pts = []
    for ndc_x, ndc_y in lm_ndc:
        img_x = (1.0 - ndc_x) / 2.0  # un-NDC to normalized image coord
        img_y = (1.0 - ndc_y) / 2.0
        # Mirror x for display on flipped frame
        px = int((1.0 - img_x) * w)
        py = int(img_y * h)
        pts.append((px, py))

    # Draw bones
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], COLOR_BONE, 2)

    # Draw joints
    finger_states = hand_data.finger_states or {}
    for i, (px, py) in enumerate(pts):
        if i in FINGERTIPS:
            fname = FINGER_TIP_TO_NAME[i]
            extended = finger_states.get(fname, False)
            color = COLOR_OPEN if extended else COLOR_CLOSED
            cv2.circle(frame, (px, py), 7, color, -1)
            cv2.circle(frame, (px, py), 7, (255, 255, 255), 1)
        else:
            cv2.circle(frame, (px, py), 4, COLOR_JOINT, -1)

    # Draw palm center
    pcx = int((1.0 - (1.0 - hand_data.palm_ndc_x) / 2.0) * w)
    pcy = int((1.0 - hand_data.palm_ndc_y) / 2.0 * h)
    cv2.circle(frame, (pcx, pcy), 10, COLOR_PALM, 2)
    cv2.drawMarker(frame, (pcx, pcy), COLOR_PALM, cv2.MARKER_CROSS, 20, 2)

    # --- Status panel ---
    is_open = hand_data.is_open_palm
    status_color = COLOR_OPEN if is_open else (0, 200, 255)

    cv2.rectangle(frame, (5, 5), (340, 180), COLOR_TEXT_BG, -1)
    cv2.rectangle(frame, (5, 5), (340, 180), status_color, 2)

    status_text = "OPEN PALM" if is_open else "CLOSED"
    cv2.putText(frame, status_text, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    # EMA confidence bar
    ema = tracker._ema_confidence
    bar_x, bar_y, bar_w, bar_h = 15, 42, 200, 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    fill_w = int(bar_w * min(ema, 1.0))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), status_color, -1)
    # Threshold line at 0.5
    thresh_x = bar_x + int(bar_w * 0.5)
    cv2.line(frame, (thresh_x, bar_y - 2), (thresh_x, bar_y + bar_h + 2), (255, 255, 255), 1)
    cv2.putText(frame, f"EMA: {ema:.3f} (thresh=0.5)", (bar_x + bar_w + 5, bar_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # Per-finger status
    y_off = 70
    for fname in ["thumb", "index", "middle", "ring", "pinky"]:
        extended = finger_states.get(fname, False)
        color = COLOR_OPEN if extended else COLOR_CLOSED
        marker = "[X]" if extended else "[ ]"
        cv2.putText(frame, f"{marker} {fname}", (15, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        y_off += 20

    # NDC coordinates
    ndc_text = f"Palm NDC: ({hand_data.palm_ndc_x:.2f}, {hand_data.palm_ndc_y:.2f})"
    cv2.putText(frame, ndc_text, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def main():
    print("=== Hand Tracker Visual Test ===")
    print("Press Q or ESC to quit.\n")

    tracker = HandTracker()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return

    fps_time = time.monotonic()
    fps_count = 0
    fps_display = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        # Process un-mirrored frame (same as camera.py does)
        hand_data = tracker.process(frame)

        # Mirror frame for display so it feels natural
        frame = cv2.flip(frame, 1)

        draw_hand_overlay(frame, hand_data, tracker)

        # FPS counter
        fps_count += 1
        elapsed = time.monotonic() - fps_time
        if elapsed >= 1.0:
            fps_display = fps_count / elapsed
            fps_count = 0
            fps_time = time.monotonic()

        cv2.putText(frame, f"FPS: {fps_display:.0f}", (frame.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Hand Tracker Test", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27:
            break

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
