"""
Pose estimator test — live webcam with skeleton + pose mapping debug.

Controls:
  Q / ESC  quit
"""

import os
import cv2
import numpy as np
import mediapipe as mp

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_lite.task")
GESTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image", "gesture")

# ── landmark indices ─────────────────────────────────────────────────────────
NOSE           = 0
LEFT_SHOULDER  = 11; RIGHT_SHOULDER = 12
LEFT_ELBOW     = 13; RIGHT_ELBOW    = 14
LEFT_WRIST     = 15; RIGHT_WRIST    = 16
LEFT_HIP       = 23; RIGHT_HIP      = 24

# Skeleton connections to draw
_CONNECTIONS = [
    # torso
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    # left arm
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    # right arm
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
]

# ── pose definitions ─────────────────────────────────────────────────────────
POSES = [
    ("praise_sun", "praise_the_sun.png"),
    ("point_down", "_point_down.png"),
    ("point_up",   "_point_up.png"),
    ("bow",        "_bow.png"),
    ("wave",       "_wave.png"),
    ("welcome",    "welcome.png"),
    ("applause",   "applause.png"),
]
POSE_NAMES = [p[0] for p in POSES]


def check_pose(name: str, lm) -> bool:
    if name == "praise_sun":
        l_high  = lm[LEFT_WRIST].y  < lm[LEFT_SHOULDER].y  - 0.05
        r_high  = lm[RIGHT_WRIST].y < lm[RIGHT_SHOULDER].y - 0.05
        span_w  = abs(lm[LEFT_WRIST].x  - lm[RIGHT_WRIST].x)
        span_sh = abs(lm[LEFT_SHOULDER].x - lm[RIGHT_SHOULDER].x)
        return l_high and r_high and span_w > span_sh * 1.3

    if name == "point_down":
        l = (lm[LEFT_ELBOW].y  > lm[LEFT_SHOULDER].y
             and lm[LEFT_WRIST].y  > lm[LEFT_ELBOW].y
             and lm[LEFT_WRIST].y  > lm[LEFT_HIP].y  + 0.05)
        r = (lm[RIGHT_ELBOW].y > lm[RIGHT_SHOULDER].y
             and lm[RIGHT_WRIST].y > lm[RIGHT_ELBOW].y
             and lm[RIGHT_WRIST].y > lm[RIGHT_HIP].y + 0.05)
        return l or r

    if name == "point_up":
        l_up = lm[LEFT_WRIST].y  < lm[NOSE].y - 0.05
        r_up = lm[RIGHT_WRIST].y < lm[NOSE].y - 0.05
        return bool(l_up ^ r_up)   # exactly one arm up

    if name == "bow":
        avg_sh = (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2
        return lm[NOSE].y > avg_sh + 0.20

    if name == "wave":
        l_r = lm[LEFT_WRIST].y  < lm[LEFT_SHOULDER].y  - 0.08
        r_r = lm[RIGHT_WRIST].y < lm[RIGHT_SHOULDER].y - 0.08
        return bool(l_r ^ r_r)     # exactly one arm raised

    if name == "welcome":
        l_lv = abs(lm[LEFT_WRIST].y  - lm[LEFT_SHOULDER].y)  < 0.12
        r_lv = abs(lm[RIGHT_WRIST].y - lm[RIGHT_SHOULDER].y) < 0.12
        span_w  = abs(lm[LEFT_WRIST].x  - lm[RIGHT_WRIST].x)
        span_sh = abs(lm[LEFT_SHOULDER].x - lm[RIGHT_SHOULDER].x)
        return l_lv and r_lv and span_w > span_sh * 1.4

    if name == "applause":
        span_sh  = abs(lm[LEFT_SHOULDER].x - lm[RIGHT_SHOULDER].x)
        wrist_d  = abs(lm[LEFT_WRIST].x    - lm[RIGHT_WRIST].x)
        together = wrist_d < span_sh * 0.5
        l_chest  = lm[LEFT_SHOULDER].y  < lm[LEFT_WRIST].y  < lm[LEFT_HIP].y
        r_chest  = lm[RIGHT_SHOULDER].y < lm[RIGHT_WRIST].y < lm[RIGHT_HIP].y
        return together and l_chest and r_chest

    return False


# ── helpers ──────────────────────────────────────────────────────────────────

def lm_px(lm_point, w, h):
    return int(lm_point.x * w), int(lm_point.y * h)


def load_gesture_icons(target_h=100):
    icons = {}
    for name, filename in POSES:
        path = os.path.join(GESTURE_DIR, filename)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[icon] missing: {filename}")
            continue
        scale = target_h / img.shape[0]
        new_w = int(img.shape[1] * scale)
        icons[name] = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
    return icons


def blend_icon(frame, icon_bgra, x, y):
    """Alpha-blend a 4-channel icon onto frame at (x, y) top-left."""
    ih, iw = icon_bgra.shape[:2]
    # Clip to frame bounds
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + iw, frame.shape[1]), min(y + ih, frame.shape[0])
    if x2 <= x1 or y2 <= y1:
        return
    roi      = frame[y1:y2, x1:x2]
    src      = icon_bgra[y1-y : y1-y + (y2-y1), x1-x : x1-x + (x2-x1)]
    alpha    = src[:, :, 3:4].astype(np.float32) / 255.0
    bgr      = src[:, :, :3].astype(np.float32)
    frame[y1:y2, x1:x2] = (bgr * alpha + roi * (1 - alpha)).astype(np.uint8)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # Build landmarker
    BaseOptions          = mp.tasks.BaseOptions
    PoseLandmarker       = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode    = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = PoseLandmarker.create_from_options(options)

    icons = load_gesture_icons(target_h=100)
    ema   = {p: 0.0 for p in POSE_NAMES}
    ts_ms = 0

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    PANEL_W  = 260          # right-side debug panel width
    BAR_MAX  = 180          # max pixel length of EMA bar

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        h, w = frame.shape[:2]
        frame = cv2.flip(frame, 1)   # mirror so it feels natural

        # ── run pose landmark detection ──────────────────────────────
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms += 33
        result = landmarker.detect_for_video(mp_img, ts_ms)

        active_pose = None   # highest-priority pose above EMA threshold

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]

            # ── update EMAs ─────────────────────────────────────────
            for name in POSE_NAMES:
                raw     = 1.0 if check_pose(name, lm) else 0.0
                ema[name] = ema[name] * 0.7 + raw * 0.3

            # First in priority order that crosses 0.5
            for name in POSE_NAMES:
                if ema[name] > 0.5:
                    active_pose = name
                    break

            # ── draw skeleton ────────────────────────────────────────
            bone_color  = (0, 220, 255) if active_pose else (180, 180, 180)
            joint_color = (255, 255, 255)

            for a, b in _CONNECTIONS:
                ax, ay = lm_px(lm[a], w, h)
                bx, by = lm_px(lm[b], w, h)
                cv2.line(frame, (ax, ay), (bx, by), bone_color, 2, cv2.LINE_AA)

            key_indices = {NOSE, LEFT_SHOULDER, RIGHT_SHOULDER,
                           LEFT_ELBOW, RIGHT_ELBOW,
                           LEFT_WRIST, RIGHT_WRIST,
                           LEFT_HIP, RIGHT_HIP}
            for i, pt in enumerate(lm):
                px, py = lm_px(pt, w, h)
                if i in key_indices:
                    cv2.circle(frame, (px, py), 5, joint_color, -1, cv2.LINE_AA)

        else:
            # No detection — decay all EMAs
            for name in POSE_NAMES:
                ema[name] *= 0.7

        # ── right-side EMA panel ─────────────────────────────────────
        panel_x = w - PANEL_W
        cv2.rectangle(frame, (panel_x, 0), (w, h), (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, 0), (w, h), (60, 60, 60), 1)

        cv2.putText(frame, "POSE DETECTOR", (panel_x + 10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 168, 78), 1, cv2.LINE_AA)
        cv2.line(frame, (panel_x + 5, 36), (w - 5, 36), (80, 80, 80), 1)

        row_h  = 100 + 10     # icon height + gap
        icon_x = panel_x + 10

        for idx, name in enumerate(POSE_NAMES):
            val      = ema[name]
            is_active = (name == active_pose)
            row_y    = 45 + idx * row_h

            # Gesture icon
            if name in icons:
                blend_icon(frame, icons[name], icon_x, row_y)
            icon_w = icons[name].shape[1] if name in icons else 0

            # Pose name label
            label_color = (0, 255, 160) if is_active else (180, 180, 180)
            display_name = name.replace("_", " ").title()
            cv2.putText(frame, display_name,
                        (icon_x + icon_w + 8, row_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, label_color, 1, cv2.LINE_AA)

            # EMA bar
            bar_w = int(val * BAR_MAX)
            bar_y = row_y + 28
            bar_color = (0, 220, 80) if val > 0.5 else (60, 160, 220)
            cv2.rectangle(frame, (icon_x + icon_w + 8, bar_y),
                          (icon_x + icon_w + 8 + BAR_MAX, bar_y + 10), (50, 50, 50), -1)
            if bar_w > 0:
                cv2.rectangle(frame, (icon_x + icon_w + 8, bar_y),
                              (icon_x + icon_w + 8 + bar_w, bar_y + 10), bar_color, -1)
            # threshold marker at 0.5
            tx = icon_x + icon_w + 8 + int(0.5 * BAR_MAX)
            cv2.line(frame, (tx, bar_y - 2), (tx, bar_y + 12), (220, 220, 60), 1)

            cv2.putText(frame, f"{val:.2f}",
                        (icon_x + icon_w + 8 + BAR_MAX + 4, bar_y + 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

        # ── active pose label (top of main viewport) ─────────────────
        if active_pose:
            label = active_pose.replace("_", " ").upper()
            cv2.putText(frame, label, (20, 42),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 220, 80), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "no pose", (20, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1, cv2.LINE_AA)

        cv2.imshow("Pose Estimator Test  [Q/ESC = quit]", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
