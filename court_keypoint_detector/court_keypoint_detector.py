# court_keypoint_detector.py
from ultralytics import YOLO
import supervision as sv
from typing import List, Optional, Any
from utils import read_stub, save_stub
import numpy as np

class CourtKeypointDetector:
    """
    Robust court-keypoint detector wrapper around a YOLO model.
    Returns list length == frames length. Each element is either:
      - an object with .xy (shape (1,N,2)) compatible with downstream code
      - or None if no keypoints detected
    """

    def __init__(self, model_path: str, batch_size: int = 20, conf: float = 0.5):
        self.model = YOLO(model_path)
        self.batch_size = batch_size
        self.conf = conf

    def _wrap_to_kpobj(self, arr):
        """
        Ensure arr is converted to an object with .xy (shape (1,N,2)).
        Returns None if arr invalid/empty.
        """
        if arr is None:
            return None
        # If already has .xy, keep as-is
        if hasattr(arr, "xy"):
            return arr
        # Convert numpy/list to array
        try:
            arr_np = np.array(arr, dtype=np.float32)
            # If shape is (N,2) turn into (1,N,2)
            if arr_np.ndim == 2 and arr_np.shape[1] == 2:
                arr_out = arr_np.reshape(1, arr_np.shape[0], 2)
            elif arr_np.ndim == 3 and arr_np.shape[2] == 2:
                arr_out = arr_np
            else:
                return None
            class KP:
                def __init__(self, xy):
                    self.xy = xy
                    # xyn placeholder for compatibility
                    self.xyn = xy / np.array([[[max(1, xy.shape[2]), max(1, xy.shape[1])]]], dtype=np.float32)
            return KP(arr_out)
        except Exception:
            return None

    def _normalize_keypoints_object(self, kp_obj: Any) -> Optional[Any]:
        """
        Normalize the model's keypoints output:
         - if it's already usable keep it
         - if it's array/list convert/wrap to object with .xy
         - else return None
        """
        if kp_obj is None:
            return None
        if hasattr(kp_obj, "xy"):
            return kp_obj
        # try to convert
        return self._wrap_to_kpobj(kp_obj)

    def get_court_keypoints(self,
                            frames: List,
                            read_from_stub: bool = False,
                            stub_path: Optional[str] = None) -> List[Optional[Any]]:
        # Try to read stub first
        if read_from_stub and stub_path:
            try:
                st = read_stub(read_from_stub, stub_path)
                if st is not None and len(st) == len(frames):
                    print(f"[DBG] CourtKeypointDetector: loaded {len(st)} keypoint frames from stub {stub_path}")
                    return st
            except Exception as e:
                print(f"[DBG] CourtKeypointDetector: read_stub failed: {e} — recomputing")

        court_keypoints: List[Optional[Any]] = []
        total = len(frames)
        if total == 0:
            return []

        for i in range(0, total, self.batch_size):
            batch_frames = frames[i:i + self.batch_size]
            try:
                results = self.model.predict(batch_frames, conf=self.conf)
            except Exception as e:
                print(f"[DBG] CourtKeypointDetector: model.predict failed on batch {i}-{i+len(batch_frames)-1}: {e}")
                court_keypoints.extend([None] * len(batch_frames))
                continue

            for res in results:
                kp = None
                if hasattr(res, "keypoints"):
                    kp_raw = res.keypoints
                    kp = self._normalize_keypoints_object(kp_raw)
                else:
                    # try to collect keypoints from supervision structures if any (robustness)
                    try:
                        # ultralytics structures sometimes provide .masks/.boxes but no keypoints -> None
                        kp = None
                    except Exception:
                        kp = None

                court_keypoints.append(kp)

        # ensure output length matches frames
        if len(court_keypoints) < total:
            court_keypoints.extend([None] * (total - len(court_keypoints)))
        elif len(court_keypoints) > total:
            court_keypoints = court_keypoints[:total]

        # Save stub (overwrite) if requested
        if stub_path:
            try:
                save_stub(stub_path, court_keypoints)
                print(f"[DBG] CourtKeypointDetector: saved stub to {stub_path}")
            except Exception as e:
                print(f"[DBG] CourtKeypointDetector: failed saving stub {stub_path}: {e}")

        return court_keypoints