# player_tracker.py
import os
import sys
import numpy as np

from ultralytics import YOLO

# keep repo utils path pattern consistent
folder = os.path.dirname(__file__)
sys.path.append(os.path.join(folder, "../"))

try:
    import supervision as sv
    _HAS_SUPERVISION = True
except Exception:
    _HAS_SUPERVISION = False

from utils import read_stub, save_stub

class PlayerTracker:
    """
    Robust PlayerTracker using Ultralytics YOLO + supervision.ByteTrack (if available).

    Public API same as before:
      - PlayerTracker(model_path)
      - get_object_tracks(frames, read_from_stub=False, stub_path=None)
    """

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        # Try to instantiate ByteTrack if supervision is present
        self.tracker = None
        if _HAS_SUPERVISION:
            try:
                self.tracker = sv.ByteTrack()
            except Exception:
                self.tracker = None

    def _detect_batch(self, frames, batch_size=20, conf=0.5):
        """
        Run YOLO predict in batches and return list of Results aligned with frames.
        """
        results = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            res_batch = self.model.predict(batch, conf=conf)
            results.extend(res_batch)
        return results

    def _extract_player_detections(self, result):
        """
        Given one ultralytics Result, return a supervision.Detections object if possible,
        or a simple list of (bbox, conf, cls_id) tuples as fallback.
        """
        # Preferred: convert to supervision.Detections
        try:
            if _HAS_SUPERVISION:
                dets = sv.Detections.from_ultralytics(result)
                return dets
        except Exception:
            pass

        # Fallback: try to extract boxes, conf, cls arrays from result.boxes
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return None

        xyxy = None
        confs = None
        cls_ids = None
        try:
            if hasattr(boxes, "xyxy"):
                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
            if hasattr(boxes, "conf"):
                confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
            if hasattr(boxes, "cls"):
                cls_ids = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)
        except Exception:
            return None

        if xyxy is None or len(xyxy) == 0:
            return None

        # return simple list of tuples to be consumed by tracking logic below
        fallback = []
        for i, b in enumerate(xyxy):
            c = float(confs[i]) if confs is not None else 0.0
            cls = int(cls_ids[i]) if cls_ids is not None else None
            fallback.append((list(map(float, b.tolist())), c, cls))
        return fallback

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Return a list of length n_frames where each element is a dict:
            { track_id: {"bbox": [x1,y1,x2,y2]} }
        Uses read_stub/save_stub for caching (same semantics as your repo).
        """
        # Try to read cached tracks
        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None and len(tracks) == len(frames):
            print("[DBG] PlayerTracker: loaded tracks from stub")
            return tracks

        # Run detector
        results = self._detect_batch(frames)

        # Prepare name-to-id mapping heuristics per model result
        # We'll get names from the first result if possible
        names = {}
        if len(results) > 0 and hasattr(results[0], "names"):
            names = getattr(results[0], "names") or {}

        # Candidate class labels to consider as 'player'
        candidate_labels = {"player", "Player", "person", "Person", "man", "woman"}
        name_to_id = {v: k for k, v in names.items()} if names else {}

        possible_player_cls_ids = []
        for lab in candidate_labels:
            if lab in name_to_id:
                possible_player_cls_ids.append(name_to_id[lab])

        # If we couldn't map anything and model has only one class, assume it's the players class
        if not possible_player_cls_ids and len(name_to_id) == 1:
            possible_player_cls_ids = [list(name_to_id.values())[0]]

        # If supervision ByteTrack is available, we'll feed Detections to it
        use_tracker = (self.tracker is not None)

        # Initialize per-frame tracks (empty dicts)
        frame_tracks = [{} for _ in range(len(results))]

        # If using supervision tracker: iterate converting each result to Detections and track
        if use_tracker:
            for i, res in enumerate(results):
                dets = self._extract_player_detections(res)
                if dets is None:
                    # nothing for this frame
                    continue
                try:
                    # Some supervision versions expect detections with boxes, confidences and class ids
                    tracked = self.tracker.update_with_detections(dets)
                except Exception:
                    # fallback to update(detections) if method name differs
                    try:
                        tracked = self.tracker.update(dets)
                    except Exception:
                        tracked = []

                # tracked might be an iterable of tuples or a special structure
                # supervision ByteTrack returns a list-like where each item contains [xyxy, score, class_id, track_id]
                for item in tracked:
                    try:
                        # try the common pattern: item[0]=bbox, item[3]=class_id or item[4]=track_id depending on version
                        if len(item) >= 5:
                            bbox = list(map(float, item[0].tolist())) if hasattr(item[0], "tolist") else list(map(float, item[0]))
                            cls_id = int(item[3])
                            track_id = int(item[4])
                        elif len(item) >= 4:
                            bbox = list(map(float, item[0].tolist())) if hasattr(item[0], "tolist") else list(map(float, item[0]))
                            cls_id = int(item[2])
                            track_id = int(item[3])
                        else:
                            continue
                    except Exception:
                        continue

                    # filter by class id if possible
                    if possible_player_cls_ids and cls_id not in possible_player_cls_ids:
                        continue

                    frame_tracks[i][track_id] = {"bbox": bbox}

        else:
            # No ByteTrack available — use simple per-frame highest-confidence per-object detection and assign temporary IDs
            next_tmp_id = 1
            for i, res in enumerate(results):
                dets = self._extract_player_detections(res)
                if dets is None:
                    continue

                # If we got supervision.Detections, iterate it
                if _HAS_SUPERVISION and isinstance(dets, sv.Detections):
                    # sync arrays
                    boxes = dets.xyxy if hasattr(dets, "xyxy") else None
                    confs = dets.confidence if hasattr(dets, "confidence") else None
                    cls_ids = dets.class_id if hasattr(dets, "class_id") else None
                    # iterate detections
                    for bbox, conf, cls in zip(boxes, confs, cls_ids):
                        cls_int = int(cls)
                        if possible_player_cls_ids and cls_int not in possible_player_cls_ids:
                            continue
                        bb = bbox.tolist() if hasattr(bbox, "tolist") else list(map(float, bbox))
                        frame_tracks[i][next_tmp_id] = {"bbox": list(map(float, bb))}
                        next_tmp_id += 1
                else:
                    # fallback list of tuples (bbox, conf, cls)
                    for bbox, conf, cls in dets:
                        cls_int = int(cls) if cls is not None else None
                        if possible_player_cls_ids and cls_int not in possible_player_cls_ids:
                            continue
                        frame_tracks[i][next_tmp_id] = {"bbox": list(map(float, bbox))}
                        next_tmp_id += 1

        # Save stub
        try:
            save_stub(stub_path, frame_tracks)
        except Exception as e:
            print(f"[DBG] PlayerTracker: failed to save stub: {e}")

        print(f"[DBG] PlayerTracker: finished. frames={len(frame_tracks)}")
        return frame_tracks