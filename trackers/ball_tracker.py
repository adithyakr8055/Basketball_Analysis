# ball_tracker.py
from ultralytics import YOLO
import numpy as np
import pandas as pd
import sys
import os

# keep repo utils path pattern consistent
folder = os.path.dirname(__file__)
sys.path.append(os.path.join(folder, "../"))

try:
    import supervision as sv
    _HAS_SUPERVISION = True
except Exception:
    _HAS_SUPERVISION = False

from utils import read_stub, save_stub

class BallTracker:
    """
    Robust BallTracker using an Ultralytics YOLO model.
    """

    def __init__(self, model_path):
        # instantiate model once
        self.model = YOLO(model_path)

    def detect_frames(self, frames, batch_size: int = 20, conf: float = 0.5):
        """
        Run the YOLO model in batches and return list of Result objects (one per frame).
        """
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            # model.predict returns a list-like of Results for the batch
            results = self.model.predict(batch, conf=conf)
            # results is iterable and aligns with frames slice
            detections.extend(results)
        return detections

    def _extract_ball_bbox_from_result(self, result):
        """
        Given a single ultralytics Result, return the chosen ball bbox as [x1,y1,x2,y2]
        using the highest-confidence Ball class. Returns None if none found.
        Works with either supervision conversions or direct result.boxes.
        """
        # Try supervision if available and result is compatible
        try:
            if _HAS_SUPERVISION:
                dets = sv.Detections.from_ultralytics(result)
                # dets.boxes is an array Nx4, dets.confidence Nx1, dets.class_id Nx1
                if len(dets) == 0:
                    return None
                names = result.names if hasattr(result, "names") else {}
                # find 'Ball' key if exists (case-sensitive in many models)
                # build inverse mapping name->id
                name_to_id = {v: k for k, v in names.items()} if names else {}
                # prefer class named 'Ball' or 'basketball' (heuristic)
                candidate_class_ids = []
                for preferred in ("Ball", "ball", "basketball", "Basketball"):
                    if preferred in name_to_id:
                        candidate_class_ids.append(name_to_id[preferred])
                # fallback: if only one class in model and it's ball-like, try it
                if not candidate_class_ids and len(names) == 1:
                    candidate_class_ids = [list(names.keys())[0]]

                # Iterate detections and pick highest confidence among candidate_class_ids
                chosen = None
                max_conf = -1
                for bbox, conf, cls in zip(dets.xyxy, dets.confidence, dets.class_id):
                    cls = int(cls)
                    if candidate_class_ids:
                        if cls not in candidate_class_ids:
                            continue
                    # if candidate list empty (we couldn't map any names), accept all and choose best with heuristics
                    if conf > max_conf:
                        max_conf = float(conf)
                        chosen = list(map(float, bbox.tolist()))
                return chosen
        except Exception:
            # supervision path failed — continue to try direct extraction below
            pass

        # Fallback: try Result.boxes (Ultralytics)
        try:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                return None
            # boxes.xyxy might be a tensor; try to get numpy
            xyxy = None
            confs = None
            cls_ids = None
            if hasattr(boxes, "xyxy"):
                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
            if hasattr(boxes, "conf"):
                confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
            if hasattr(boxes, "cls"):
                cls_ids = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)

            if xyxy is None or len(xyxy) == 0:
                return None

            # try to identify ball class id from result.names
            names = getattr(result, "names", {})
            name_to_id = {v: k for k, v in names.items()} if names else {}
            candidate_class_ids = []
            for preferred in ("Ball", "ball", "basketball", "Basketball"):
                if preferred in name_to_id:
                    candidate_class_ids.append(name_to_id[preferred])
            if not candidate_class_ids and cls_ids is not None and len(np.unique(cls_ids)) == 1:
                candidate_class_ids = [int(np.unique(cls_ids)[0])]

            chosen = None
            max_conf = -1
            for idx, box in enumerate(xyxy):
                conf = float(confs[idx]) if confs is not None else 0.0
                cls = int(cls_ids[idx]) if cls_ids is not None else None
                if candidate_class_ids:
                    if cls not in candidate_class_ids:
                        continue
                if conf > max_conf:
                    max_conf = conf
                    chosen = list(map(float, box.tolist()))
            return chosen
        except Exception:
            return None

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Run detection over frames and return a list (len = n_frames) of dicts like:
            [{1: {"bbox":[x1,y1,x2,y2]}}, {}, {1: {...}}, ...]
        Uses read_stub/save_stub for caching (same semantics as your repo).
        """
        # Try to read cached tracks
        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None and len(tracks) == len(frames):
            print("[DBG] BallTracker: loaded tracks from stub")
            return tracks

        # Run detector
        results = self.detect_frames(frames)

        tracks = []
        for i, res in enumerate(results):
            chosen_bbox = self._extract_ball_bbox_from_result(res)
            if chosen_bbox is None:
                tracks.append({})  # no ball detected this frame
            else:
                # normalize bbox to list of floats [x1,y1,x2,y2]
                tracks.append({1: {"bbox": chosen_bbox}})
        # Save stub (best-effort)
        try:
            save_stub(stub_path, tracks)
        except Exception as e:
            print(f"[DBG] BallTracker: failed to save stub: {e}")

        return tracks

    def remove_wrong_detections(self, ball_positions, maximum_allowed_distance: float = 25.0):
        """
        Remove detections that jump too far from the last good detection.
        ball_positions: list of dicts like [{1: {"bbox": [...] }}, {} ...]
        Works in-place and returns the filtered list.
        Uses bbox center for distance measurement.
        """
        last_good_idx = -1
        last_good_center = None

        for i in range(len(ball_positions)):
            info = ball_positions[i].get(1, {})
            bbox = info.get("bbox", [])
            if not bbox or len(bbox) < 4:
                continue

            # compute center
            x1, y1, x2, y2 = map(float, bbox[:4])
            center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

            if last_good_idx == -1:
                last_good_idx = i
                last_good_center = center
                continue

            frame_gap = i - last_good_idx
            allowed = maximum_allowed_distance * frame_gap
            dist = np.linalg.norm(center - last_good_center)
            if dist > allowed:
                # mark as missing
                ball_positions[i] = {}
            else:
                last_good_idx = i
                last_good_center = center

        return ball_positions

    def interpolate_ball_positions(self, ball_positions):
        """
        Turn list of dicts into DataFrame, interpolate missing values and return list of dicts.
        Input: [{1: {"bbox":[...] }}, {}, ...]
        Output: [{1: {"bbox":[...] }}, ...] with interpolated bboxes (floats)
        """
        # Extract bboxes as rows (x1,y1,x2,y2) or NaNs
        rows = []
        for item in ball_positions:
            info = item.get(1, {})
            bbox = info.get("bbox", [])
            if bbox and len(bbox) >= 4:
                rows.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])
            else:
                rows.append([np.nan, np.nan, np.nan, np.nan])

        df = pd.DataFrame(rows, columns=["x1", "y1", "x2", "y2"])

        # if all NaN, return same empty structure
        if df.isna().all().all():
            return [{ } for _ in rows]

        # Interpolate and fill both directions; final fallback fill with nearest
        df = df.interpolate(method='linear', limit_direction='both', axis=0)
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0.0)

        out = []
        for row in df.to_numpy().tolist():
            # convert to floats
            try:
                x1, y1, x2, y2 = map(float, row)
                out.append({1: {"bbox": [x1, y1, x2, y2]}})
            except Exception:
                out.append({})

        return out