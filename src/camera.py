import threading
import cv2
import numpy as np

from hand_tracker import HandTracker, HandData
from pose_tracker import PoseTracker, PoseData

CAPTURE_W, CAPTURE_H = 80, 60


class Camera:
    def __init__(self, device=0, enable_hand=True, enable_pose=True):
        self._device = device
        self._cap = None
        self._ready = False

        self._lock = threading.Lock()
        self._brightness = np.zeros((CAPTURE_H, CAPTURE_W), dtype=np.float32)
        self._motion = np.zeros((CAPTURE_H, CAPTURE_W), dtype=np.float32)
        self._preview = np.zeros((CAPTURE_H * 2, CAPTURE_W * 2, 3), dtype=np.uint8)
        self._avg_motion = 0.0
        self._prev_gray = None
        self._running = True

        self._enable_hand = enable_hand
        self._hand_tracker = HandTracker() if enable_hand else None
        self._hand_data = HandData()
        self._hand_ema = 0.0

        self._enable_pose = enable_pose
        self._pose_tracker = PoseTracker() if enable_pose else None
        self._pose_data = PoseData()

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    @property
    def ready(self):
        with self._lock:
            return self._ready

    @property
    def hand_enabled(self) -> bool:
        with self._lock:
            return self._enable_hand

    @hand_enabled.setter
    def hand_enabled(self, value: bool):
        with self._lock:
            self._enable_hand = value
            if value and self._hand_tracker is None:
                self._hand_tracker = HandTracker()

    @property
    def pose_enabled(self) -> bool:
        with self._lock:
            return self._enable_pose

    @pose_enabled.setter
    def pose_enabled(self, value: bool):
        with self._lock:
            self._enable_pose = value
            if value and self._pose_tracker is None:
                self._pose_tracker = PoseTracker()

    def _capture_loop(self):
        # Open camera in background thread (avoids blocking main thread)
        self._cap = cv2.VideoCapture(self._device, cv2.CAP_DSHOW)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                continue

            # Only run enabled trackers
            with self._lock:
                run_hand = self._enable_hand
                run_pose = self._enable_pose

            hand_data = self._hand_tracker.process(frame) if (run_hand and self._hand_tracker) else HandData()
            pose_data = self._pose_tracker.process(frame) if (run_pose and self._pose_tracker) else PoseData()

            small = cv2.resize(frame, (CAPTURE_W, CAPTURE_H), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            brightness = gray.astype(np.float32) / 255.0

            if self._prev_gray is not None:
                diff = np.abs(gray.astype(np.float32) - self._prev_gray.astype(np.float32)) / 255.0
            else:
                diff = np.zeros_like(brightness)
            self._prev_gray = gray

            avg_m = float(np.mean(diff))

            preview = cv2.resize(frame, (CAPTURE_W * 2, CAPTURE_H * 2), interpolation=cv2.INTER_AREA)
            preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

            with self._lock:
                self._brightness = brightness
                self._motion = diff
                self._avg_motion = avg_m
                self._preview = preview
                self._hand_data = hand_data
                self._hand_ema = getattr(self._hand_tracker, '_ema_confidence', 0.0) if self._hand_tracker else 0.0
                self._pose_data = pose_data
                self._ready = True

    def get_data(self):
        with self._lock:
            return self._brightness.copy(), self._motion.copy(), self._avg_motion

    def get_hand_data(self) -> HandData:
        with self._lock:
            return self._hand_data

    def get_hand_ema(self) -> float:
        with self._lock:
            return self._hand_ema

    def get_pose_data(self) -> PoseData:
        with self._lock:
            return self._pose_data

    def get_preview(self):
        with self._lock:
            return self._preview.copy()

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)
        if self._hand_tracker:
            self._hand_tracker.close()
        if self._pose_tracker:
            self._pose_tracker.close()
        if self._cap is not None:
            self._cap.release()
