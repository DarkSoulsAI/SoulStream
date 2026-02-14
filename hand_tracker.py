import os
from dataclasses import dataclass

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


@dataclass
class HandData:
    detected: bool = False
    is_open_palm: bool = False
    palm_ndc_x: float = 0.0
    palm_ndc_y: float = 0.0
    landmarks: list = None  # 21 (ndc_x, ndc_y) tuples when detected
    # Debug: per-finger extension state
    finger_states: dict = None


try:
    import mediapipe as mp

    class HandTracker:
        def __init__(self):
            BaseOptions = mp.tasks.BaseOptions
            HandLandmarker = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=VisionRunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = HandLandmarker.create_from_options(options)
            self._ema_confidence = 0.0
            self._frame_ts_ms = 0

        def process(self, frame_bgr) -> HandData:
            import cv2

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            self._frame_ts_ms += 33  # ~30fps increment
            result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)

            if not result.hand_landmarks:
                self._ema_confidence = self._ema_confidence * 0.7
                return HandData(detected=False)

            lm = result.hand_landmarks[0]  # list of 21 NormalizedLandmark

            # Open palm detection: all 5 fingers extended
            # Fingers: tip y < mcp y (image coords, lower y = higher on screen)
            finger_states = {}

            fingers_extended = (
                lm[8].y < lm[5].y    # Index: tip(8) above MCP(5)
                and lm[12].y < lm[9].y   # Middle: tip(12) above MCP(9)
                and lm[16].y < lm[13].y  # Ring: tip(16) above MCP(13)
                and lm[20].y < lm[17].y  # Pinky: tip(20) above MCP(17)
            )

            finger_states["index"] = lm[8].y < lm[5].y
            finger_states["middle"] = lm[12].y < lm[9].y
            finger_states["ring"] = lm[16].y < lm[13].y
            finger_states["pinky"] = lm[20].y < lm[17].y

            # Thumb: tip(4) x-distance from wrist(0) > MCP(2) x-distance from wrist
            thumb_tip_dist = abs(lm[4].x - lm[0].x)
            thumb_mcp_dist = abs(lm[2].x - lm[0].x)
            thumb_extended = thumb_tip_dist > thumb_mcp_dist
            finger_states["thumb"] = thumb_extended

            raw = 1.0 if (fingers_extended and thumb_extended) else 0.0
            self._ema_confidence = self._ema_confidence * 0.7 + raw * 0.3
            is_open = self._ema_confidence > 0.5

            # Palm center: average of WRIST(0) and MIDDLE_MCP(9)
            cx = (lm[0].x + lm[9].x) / 2.0
            cy = (lm[0].y + lm[9].y) / 2.0

            # NDC conversion matching camera.py mirrored convention
            palm_ndc_x = 1.0 - cx * 2.0
            palm_ndc_y = 1.0 - cy * 2.0

            # Convert all 21 landmarks to NDC
            landmarks = []
            for l in lm:
                landmarks.append((1.0 - l.x * 2.0, 1.0 - l.y * 2.0))

            return HandData(
                detected=True,
                is_open_palm=is_open,
                palm_ndc_x=palm_ndc_x,
                palm_ndc_y=palm_ndc_y,
                landmarks=landmarks,
                finger_states=finger_states,
            )

        def close(self):
            self._landmarker.close()

except (ImportError, Exception) as e:
    print(f"[HandTracker] MediaPipe unavailable ({e}), hand tracking disabled.")

    class HandTracker:
        def __init__(self):
            pass

        def process(self, frame_bgr) -> HandData:
            return HandData()

        def close(self):
            pass
