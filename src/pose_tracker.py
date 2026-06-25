import os
from dataclasses import dataclass, field
from typing import Optional

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
MODEL_PATH = os.path.join(_ROOT_DIR, "pose_landmarker_lite.task")

# MediaPipe Pose landmark indices
_NOSE           = 0
_LEFT_SHOULDER  = 11
_RIGHT_SHOULDER = 12
_LEFT_ELBOW     = 13
_RIGHT_ELBOW    = 14
_LEFT_WRIST     = 15
_RIGHT_WRIST    = 16
_LEFT_HIP       = 23
_RIGHT_HIP      = 24

# Evaluated in priority order — first EMA that crosses 0.5 wins
_POSE_PRIORITY = [
    'praise_sun',   # both arms in Y above head
    'point_down',   # one arm straight down below hip
    'point_up',     # exactly one arm raised above nose
    'bow',          # upper body bent forward
    'wave',         # exactly one arm raised above shoulder
    'welcome',      # T-pose, both arms spread at shoulder level
    'applause',     # both hands together at chest
]


@dataclass
class PoseData:
    detected: bool = False
    pose: Optional[str] = None
    left_wrist_ndc: tuple = (0.0, 0.0)
    right_wrist_ndc: tuple = (0.0, 0.0)
    landmarks: list = field(default_factory=list)   # 33 (ndc_x, ndc_y) tuples


try:
    import mediapipe as mp

    class PoseTracker:
        def __init__(self):
            if not os.path.exists(MODEL_PATH):
                print(f"[PoseTracker] Model not found: {MODEL_PATH}")
                print("  Download: https://storage.googleapis.com/mediapipe-models/"
                      "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
                self._landmarker = None
                self._ema = {p: 0.0 for p in _POSE_PRIORITY}
                self._frame_ts_ms = 0
                return

            BaseOptions = mp.tasks.BaseOptions
            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=VisionRunningMode.VIDEO,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            try:
                self._landmarker = PoseLandmarker.create_from_options(options)
            except Exception as e:
                print(f"[PoseTracker] Could not create pose landmarker ({e}), pose detection disabled.")
                self._landmarker = None
                self._frame_ts_ms = 0
                self._ema = {p: 0.0 for p in _POSE_PRIORITY}
                return
            self._frame_ts_ms = 0
            self._ema = {p: 0.0 for p in _POSE_PRIORITY}
            print("[PoseTracker] Ready — 7 poses: " + ", ".join(_POSE_PRIORITY))

        def process(self, frame_bgr) -> PoseData:
            if self._landmarker is None:
                return PoseData()

            import cv2
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            self._frame_ts_ms += 33
            result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)

            if not result.pose_landmarks:
                for p in _POSE_PRIORITY:
                    self._ema[p] *= 0.7
                return PoseData(detected=False)

            lm = result.pose_landmarks[0]

            # Update each EMA from raw boolean check
            checks = {
                'praise_sun': self._check_praise_sun(lm),
                'point_down': self._check_point_down(lm),
                'point_up':   self._check_point_up(lm),
                'bow':        self._check_bow(lm),
                'wave':       self._check_wave(lm),
                'welcome':    self._check_welcome(lm),
                'applause':   self._check_applause(lm),
            }
            for p in _POSE_PRIORITY:
                self._ema[p] = self._ema[p] * 0.7 + checks[p] * 0.3

            # First pose in priority order that exceeds threshold wins
            pose = None
            for p in _POSE_PRIORITY:
                if self._ema[p] > 0.5:
                    pose = p
                    break

            # NDC: mirror x to match hand_tracker.py convention
            landmarks_ndc = [(1.0 - l.x * 2.0, 1.0 - l.y * 2.0) for l in lm]
            lw = landmarks_ndc[_LEFT_WRIST]
            rw = landmarks_ndc[_RIGHT_WRIST]

            return PoseData(
                detected=True,
                pose=pose,
                left_wrist_ndc=lw,
                right_wrist_ndc=rw,
                landmarks=landmarks_ndc,
            )

        # ── pose checks (return 1.0 or 0.0) ──────────────────────────

        def _check_praise_sun(self, lm) -> float:
            """Both wrists raised above shoulders + spread wider than 1.3× shoulder span."""
            l_high = lm[_LEFT_WRIST].y < lm[_LEFT_SHOULDER].y - 0.05
            r_high = lm[_RIGHT_WRIST].y < lm[_RIGHT_SHOULDER].y - 0.05
            wrist_span = abs(lm[_LEFT_WRIST].x - lm[_RIGHT_WRIST].x)
            shoulder_span = abs(lm[_LEFT_SHOULDER].x - lm[_RIGHT_SHOULDER].x)
            spread = wrist_span > shoulder_span * 1.3
            return 1.0 if (l_high and r_high and spread) else 0.0

        def _check_point_down(self, lm) -> float:
            """One arm straight down, wrist clearly below hip."""
            l_down = (lm[_LEFT_ELBOW].y > lm[_LEFT_SHOULDER].y
                      and lm[_LEFT_WRIST].y > lm[_LEFT_ELBOW].y
                      and lm[_LEFT_WRIST].y > lm[_LEFT_HIP].y + 0.05)
            r_down = (lm[_RIGHT_ELBOW].y > lm[_RIGHT_SHOULDER].y
                      and lm[_RIGHT_WRIST].y > lm[_RIGHT_ELBOW].y
                      and lm[_RIGHT_WRIST].y > lm[_RIGHT_HIP].y + 0.05)
            return 1.0 if (l_down or r_down) else 0.0

        def _check_point_up(self, lm) -> float:
            """Exactly one wrist raised above nose (not both — that's praise_sun)."""
            l_up = lm[_LEFT_WRIST].y < lm[_NOSE].y - 0.05
            r_up = lm[_RIGHT_WRIST].y < lm[_NOSE].y - 0.05
            return 1.0 if (l_up ^ r_up) else 0.0   # XOR: one but not both

        def _check_bow(self, lm) -> float:
            """Upper body bent forward: nose drops significantly below shoulder level."""
            avg_shoulder_y = (lm[_LEFT_SHOULDER].y + lm[_RIGHT_SHOULDER].y) / 2
            return 1.0 if lm[_NOSE].y > avg_shoulder_y + 0.20 else 0.0

        def _check_wave(self, lm) -> float:
            """Exactly one wrist raised above its shoulder, other arm relaxed."""
            l_raised = lm[_LEFT_WRIST].y < lm[_LEFT_SHOULDER].y - 0.08
            r_raised = lm[_RIGHT_WRIST].y < lm[_RIGHT_SHOULDER].y - 0.08
            return 1.0 if (l_raised ^ r_raised) else 0.0   # XOR

        def _check_welcome(self, lm) -> float:
            """T-pose: both wrists at shoulder height (±12%), spread ≥1.4× shoulder span."""
            l_level = abs(lm[_LEFT_WRIST].y - lm[_LEFT_SHOULDER].y) < 0.12
            r_level = abs(lm[_RIGHT_WRIST].y - lm[_RIGHT_SHOULDER].y) < 0.12
            wrist_span = abs(lm[_LEFT_WRIST].x - lm[_RIGHT_WRIST].x)
            shoulder_span = abs(lm[_LEFT_SHOULDER].x - lm[_RIGHT_SHOULDER].x)
            spread = wrist_span > shoulder_span * 1.4
            return 1.0 if (l_level and r_level and spread) else 0.0

        def _check_applause(self, lm) -> float:
            """Both wrists together (< 0.5× shoulder span) at chest level."""
            shoulder_span = abs(lm[_LEFT_SHOULDER].x - lm[_RIGHT_SHOULDER].x)
            wrist_dist = abs(lm[_LEFT_WRIST].x - lm[_RIGHT_WRIST].x)
            together = wrist_dist < shoulder_span * 0.5
            l_chest = lm[_LEFT_WRIST].y > lm[_LEFT_SHOULDER].y and lm[_LEFT_WRIST].y < lm[_LEFT_HIP].y
            r_chest = lm[_RIGHT_WRIST].y > lm[_RIGHT_SHOULDER].y and lm[_RIGHT_WRIST].y < lm[_RIGHT_HIP].y
            return 1.0 if (together and l_chest and r_chest) else 0.0

        def close(self):
            if self._landmarker is not None:
                self._landmarker.close()

except (ImportError, Exception) as e:
    print(f"[PoseTracker] Unavailable ({e}), pose detection disabled.")

    class PoseTracker:
        def __init__(self): pass
        def process(self, frame_bgr) -> PoseData: return PoseData()
        def close(self): pass
