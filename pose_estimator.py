# pose_estimator.py
import cv2
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

# try import mediapipe; fail gracefully
try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except Exception:
    mp = None
    _HAS_MEDIAPIPE = False

# landmark names for reference (MediaPipe Pose)
_MP_LANDMARKS = None
if _HAS_MEDIAPIPE:
    _MP_LANDMARKS = mp.solutions.pose.PoseLandmark

def _landmark_name(idx: int) -> str:
    if not _HAS_MEDIAPIPE:
        return str(idx)
    try:
        return _MP_LANDMARKS(idx).name
    except Exception:
        return str(idx)

class PoseEstimator:
    """
    Lightweight, defensive wrapper around MediaPipe Pose.
    Methods:
      - process_frames(frames) -> List[Optional[Dict[str, Tuple[x_px,y_px,vis]]]]
    """

    def __init__(self,
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 use_gpu: bool = False):
        if not _HAS_MEDIAPIPE:
            print("[WARN] PoseEstimator: mediapipe not available; pose methods will raise on use.")
        self.model_complexity = int(model_complexity)
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.use_gpu = bool(use_gpu)

    def process_frames(self, frames: List[Any]) -> List[Optional[Dict[str, Tuple[float, float, float]]]]:
        """
        Process list of BGR frames. Returns list of same length. Each element:
          - None (if no landmarks found or frame invalid) or
          - dict {landmark_name: (x_px, y_px, visibility)}
        Coordinates returned are image pixel coordinates (int/float).
        """
        n = len(frames) if frames is not None else 0
        out: List[Optional[Dict[str, Tuple[float,float,float]]]] = [None] * n
        if n == 0:
            return out
        if not _HAS_MEDIAPIPE:
            print("[WARN] PoseEstimator: mediapipe not installed")
            return out

        # initialize mediapipe pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False,
                            model_complexity=self.model_complexity,
                            enable_segmentation=False,
                            min_detection_confidence=self.min_detection_confidence,
                            min_tracking_confidence=self.min_tracking_confidence)
        try:
            for i, f in enumerate(frames):
                try:
                    if f is None:
                        out[i] = None
                        continue
                    # mediapipe expects RGB
                    if isinstance(f, np.ndarray):
                        img_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    else:
                        out[i] = None
                        continue
                    res = pose.process(img_rgb)
                    if not res or not res.pose_landmarks:
                        out[i] = None
                        continue
                    h, w = f.shape[:2]
                    lm_map = {}
                    for idx, lm in enumerate(res.pose_landmarks.landmark):
                        name = _landmark_name(idx)
                        x_px = float(lm.x * w)
                        y_px = float(lm.y * h)
                        vis = float(getattr(lm, "visibility", 0.0))
                        lm_map[name] = (x_px, y_px, vis)
                    out[i] = lm_map
                except Exception as e:
                    # single-frame failure -> continue
                    print(f"[DBG] PoseEstimator: frame {i} failed: {e}")
                    out[i] = None
        finally:
            try:
                pose.close()
            except Exception:
                pass
        return out