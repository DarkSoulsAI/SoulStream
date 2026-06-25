import numpy as np

import hand_tracker
import pose_tracker


def test_hand_tracker_missing_model_degrades_gracefully(monkeypatch):
    monkeypatch.setattr(hand_tracker, "MODEL_PATH", "__definitely_missing_hand.task")

    tracker = hand_tracker.HandTracker()
    data = tracker.process(np.zeros((4, 4, 3), dtype=np.uint8))
    tracker.close()

    assert data.detected is False


def test_pose_tracker_missing_model_degrades_gracefully(monkeypatch):
    monkeypatch.setattr(pose_tracker, "MODEL_PATH", "__definitely_missing_pose.task")

    tracker = pose_tracker.PoseTracker()
    data = tracker.process(np.zeros((4, 4, 3), dtype=np.uint8))
    tracker.close()

    assert data.detected is False
