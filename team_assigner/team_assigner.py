"""
TeamAssigner
Assign players to teams using either CLIP (preferred) or a fast k-means color heuristic (fallback).

Public API (keeps compatibility with your pipeline):
    get_player_teams_across_frames(video_frames, player_tracks, read_from_stub=False, stub_path=None)

Notes:
- If transformers + CLIP are installed and available, CLIP will be used unless `fast_mode=True`.
- If CLIP is unavailable or fails to load, the code falls back to a k-means color heuristic.
- The function caches results (using your existing read_stub/save_stub functions) if a `stub_path` is provided.
"""

import os
import sys
from typing import Tuple, Dict, List, Optional, Any
from collections import defaultdict, Counter

import numpy as np
import cv2
from PIL import Image

# Try to import CLIP + torch lazily. If missing, we fallback to fast kmeans method.
try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    _HAS_CLIP = True
except Exception:
    _HAS_CLIP = False

# Ensure repo utils are importable (keeps same repo layout).
folder = os.path.dirname(__file__)
sys.path.append(os.path.join(folder, "../"))  # noqa: E402

# Assuming your repository structure makes 'utils' available for import
try:
    from utils import read_stub, save_stub  # your existing utils (pickle stubs)
except ImportError:
    print("[ERROR] Could not import read_stub/save_stub from utils. Check your path setup.")
    # Define dummy functions to prevent crash if running locally without full repo setup
    def read_stub(*args, **kwargs): return None
    def save_stub(*args, **kwargs): pass


def _kmeans_color_label(image_crop: Optional[np.ndarray], k: int = 2) -> str:
    """
    Fallback labeling using k-means color clustering on the player crop.
    Returns either "white shirt" or "dark blue shirt" (to match default labels),
    or "unknown" if the crop is invalid.
    """
    if image_crop is None or image_crop.size == 0:
        return "unknown"
    try:
        # Ensure we have a 3-channel BGR image
        if len(image_crop.shape) == 2:
            image_crop = cv2.cvtColor(image_crop, cv2.COLOR_GRAY2BGR)
        h, w = image_crop.shape[:2]
        data = image_crop.reshape((-1, 3)).astype(np.float32)

        # If image too small, resize to reduce noise
        if h * w < 100:
            image_crop = cv2.resize(image_crop, (64, 128), interpolation=cv2.INTER_LINEAR)
            data = image_crop.reshape((-1, 3)).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, flags)
        counts = np.bincount(labels.flatten(), minlength=k)
        dominant = centers[np.argmax(counts)]  # BGR center

        # perceived brightness from RGB
        b, g, r = float(dominant[0]), float(dominant[1]), float(dominant[2])
        brightness = 0.299 * r + 0.587 * g + 0.114 * b

        return "white shirt" if brightness > 100 else "dark blue shirt"
    except Exception:
        return "unknown"


class TeamAssigner:
    """
    Assign players to teams using CLIP (if available) or a k-means color heuristic.

    Args:
        team_1_class_name: textual label representing team 1 (default "white shirt")
        team_2_class_name: textual label representing team 2 (default "dark blue shirt")
        fast_mode: if True, skip CLIP entirely and always use k-means heuristic (good offline/fast)
        device: "cpu" or "cuda" if using CLIP; if None auto-detects
    """

    def __init__(
        self,
        team_1_class_name: str = "white shirt",
        team_2_class_name: str = "dark blue shirt",
        fast_mode: bool = False,
        device: Optional[str] = None,
    ):
        self.team_1_class_name = team_1_class_name
        self.team_2_class_name = team_2_class_name
        self.player_team_dict: Dict[int, int] = {}
        self.team_colors: Dict[int, Tuple[int, int, int]] = {}
        self.fast_mode = bool(fast_mode)

        # device selection for CLIP
        if device is not None:
            self.device = device
        else:
            if _HAS_CLIP:
                try:
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    self.device = "cpu"
            else:
                self.device = "cpu"

        # CLIP objects (loaded lazily)
        self.model: Optional[Any] = None
        self.processor: Optional[Any] = None
        self._model_loaded = False

    def load_model(self) -> bool:
        """
        Attempt to load CLIP model + processor.
        Returns True if loaded successfully.
        """
        if self.fast_mode:
            print("[DBG] TeamAssigner: fast_mode enabled — skipping CLIP model load")
            return False

        if not _HAS_CLIP:
            print("[DBG] TeamAssigner: transformers/CLIP not available — will use fallback.")
            return False

        if self._model_loaded:
            return True

        try:
            print("[DBG] TeamAssigner: loading CLIP model (may take time)...")
            # Use the same model as original code if available
            self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
            self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
            # Move model to device if possible
            try:
                self.model.to(self.device)
            except Exception:
                pass
            self._model_loaded = True
            print(f"[DBG] TeamAssigner: CLIP loaded on device={self.device}")
            return True
        except Exception as e:
            print(f"[DBG] TeamAssigner: failed to load CLIP model: {e} — falling back to kmeans")
            self._model_loaded = False
            return False

    def _crop_safe(self, frame: Optional[np.ndarray], bbox: Optional[List[float]], player_id: int = -1, frame_num: int = -1) -> Optional[np.ndarray]:
        """
        Safely crop bbox from frame. Returns None if invalid.
        Ensures bounding box clipping inside the frame and minimal crop size, with debug logging.
        """
        if frame is None or bbox is None:
            if player_id != -1 and frame_num != -1:
                # Log when frame or bbox is explicitly None
                print(f"[DBG] TeamAssigner: crop invalid (frame/bbox is None) for player {player_id} frame {frame_num}")
            return None
        try:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                if player_id != -1 and frame_num != -1:
                    print(f"[DBG] TeamAssigner: crop invalid (bbox zero-size) for player {player_id} frame {frame_num}")
                return None
            crop = frame[y1:y2, x1:x2]
            # If crop very small, resize to a reasonable size for processing
            ch, cw = crop.shape[:2]
            
            # Injecting the requested debug logging based on crop size
            if ch < 20 or cw < 20:
                if player_id != -1 and frame_num != -1:
                    print(f"[DBG] TeamAssigner: crop too small ({crop.shape[0]}x{crop.shape[1]}) for player {player_id} frame {frame_num}")
                crop = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_LINEAR)
            
            return crop
        except Exception as e:
            if player_id != -1 and frame_num != -1:
                print(f"[DBG] TeamAssigner: _crop_safe exception for player {player_id} frame {frame_num}: {e}")
            return None

    def get_player_color_clip(self, frame: np.ndarray, bbox: List[float], player_id: int = -1, frame_num: int = -1) -> str:
        """
        Use CLIP to classify the player's crop between the two class names.
        Returns one of the class labels or "unknown".
        """
        if not self._model_loaded:
            if not self.load_model():
                return "unknown"

        # Pass context for logging
        crop = self._crop_safe(frame, bbox, player_id, frame_num)
        if crop is None:
            return "unknown"

        try:
            # convert BGR -> RGB PIL
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            classes = [self.team_1_class_name, self.team_2_class_name]
            inputs = self.processor(text=classes, images=pil, return_tensors="pt", padding=True)
            # Move tensors to device selected for model if possible
            if hasattr(inputs, "to"):
                try:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                except Exception:
                    pass
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image  # shape (1, n_classes)
            probs = logits.softmax(dim=1)
            idx = int(probs.argmax(dim=1)[0])
            return classes[idx]
        except Exception as e:
            print(f"[DBG] TeamAssigner: CLIP classification error: {e}")
            return "unknown"

    def get_player_color(self, frame: np.ndarray, bbox: List[float], player_id: int = -1, frame_num: int = -1) -> str:
        """
        Public method that returns a textual class for the player's crop.
        Uses CLIP when available and not in fast_mode; otherwise k-means fallback.
        """
        crop = self._crop_safe(frame, bbox, player_id, frame_num)
        
        if self.fast_mode:
            # Pass crop directly to k-means to avoid re-cropping
            return _kmeans_color_label(crop)

        # try CLIP first (if available) - passing context
        label = self.get_player_color_clip(frame, bbox, player_id, frame_num)
        if label is None or label == "unknown":
            # fallback to kmeans - passing already cropped image
            label = _kmeans_color_label(crop)
        return label

    def get_player_team(self, frame: np.ndarray, player_bbox: List[float], player_id: int) -> int:
        """
        Return team id (1 or 2) for a single player, using cached mapping if possible.
        Defaults to team 2 if classification uncertain.
        
        NOTE: This method is primarily for compatibility; the majority-vote logic is preferred.
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Note: frame_num is not available here, so we skip passing it to get_player_color
        label = self.get_player_color(frame, player_bbox, player_id=player_id)
        team_id = 2
        if label == self.team_1_class_name:
            team_id = 1
        elif label == "unknown":
            # fallback to brightness heuristic
            crop = self._crop_safe(frame, player_bbox, player_id=player_id)
            if crop is not None:
                b, g, r = cv2.mean(crop)[:3]
                brightness = 0.299 * r + 0.587 * g + 0.114 * b
                if brightness > 100:
                    team_id = 1
        # cache this mapping for the player id (resets periodically in main pipeline)
        self.player_team_dict[player_id] = team_id
        return team_id

    def get_player_teams_across_frames(
        self,
        video_frames: List[np.ndarray],
        player_tracks: List[Dict[int, Dict]],
        read_from_stub: bool = False,
        stub_path: Optional[str] = None,
        min_votes: int = 3
    ) -> List[Dict[int, int]]:
        """
        New approach:
        - For every player occurrence across frames, collect label votes ("team1","team2","unknown")
        - After processing frames, compute a stable team id for each player by majority vote
        - Return per-frame mapping where every player's team is the stable team id (so visualization is stable)
        """
        
        # Try to load per-player stable mapping from stub if available
        stable_map = {}
        if stub_path:
            try:
                # The stub now saves the stable_map dict, not the per-frame list
                st = read_stub(read_from_stub, stub_path)
                if st is not None and isinstance(st, dict):
                    stable_map = st
                    print(f"[DBG] TeamAssigner: loaded stable_map from stub with {len(stable_map)} players")
            except Exception:
                pass

        n_frames = len(video_frames)
        
        # If stable_map exists, apply to all frames
        if stable_map and len(stable_map) > 0 and player_tracks:
            out = []
            for frame_tracks in player_tracks:
                frame_map = {}
                for pid_raw in (frame_tracks or {}).keys():
                    try:
                        pid = int(pid_raw)
                    except Exception:
                        pid = pid_raw
                    # Use stable map, default to Team 2 if player ID is new or missing
                    frame_map[pid] = stable_map.get(pid, 2) 
                out.append(frame_map)
            # Ensure output length matches video frames
            if len(out) == n_frames:
                 return out
            else:
                 print(f"[WARN] TeamAssigner: Loaded stable map but player_tracks length mismatch ({len(player_tracks)} vs {n_frames}). Recalculating.")


        # Ensure model loaded if available (CLIP)
        if not self.fast_mode and _HAS_CLIP:
            self.load_model()
            
        # gather votes
        votes = defaultdict(list)  # pid -> list of team ids (1 or 2 or 0 unknown)
        
        for fi, frame_tracks in enumerate(player_tracks or []):
            if not frame_tracks:
                continue
            # Safety check for frame index
            if fi >= n_frames:
                 print(f"[WARN] Frame index {fi} out of bounds for video frames. Stopping vote collection.")
                 break
            frame = video_frames[fi]
            
            for pid_raw, pdata in (frame_tracks or {}).items():
                try:
                    pid = int(pid_raw)
                except Exception:
                    # Use raw ID if conversion fails
                    pid = pid_raw 
                bbox = pdata.get("bbox") if isinstance(pdata, dict) else None
                if bbox is None:
                    continue
                
                # Get label (CLIP or K-means) - passing frame context for debugging
                label = self.get_player_color(frame, bbox, pid, fi)
                
                # map label to team id
                if label == self.team_1_class_name:
                    tid = 1
                elif label == self.team_2_class_name:
                    tid = 2
                else:
                    # unknown -> fallback heuristic brightness (same as get_player_team)
                    crop = self._crop_safe(frame, bbox)
                    if crop is not None and crop.size > 0:
                        b,g,r = cv2.mean(crop)[:3]
                        brightness = 0.299*r + 0.587*g + 0.114*b
                        tid = 1 if brightness > 100 else 2
                    else:
                        tid = 0 # 0 for truly uncertain/missing data
                
                votes[pid].append(tid)

        # build stable_map by majority vote, require min_votes appearances
        stable_map = {}
        for pid, arr in votes.items():
            total_votes = len(arr)
            if total_votes < min_votes:
                # not enough evidence; skip (will default to team 2 later)
                print(f"[DBG] TeamAssigner: Player {pid} skipped - only {total_votes} votes (min={min_votes})")
                continue
            
            cnt = Counter(arr)
            
            # Simple majority logic:
            count_team_1 = cnt.get(1, 0)
            count_team_2 = cnt.get(2, 0)
            
            if count_team_1 > count_team_2:
                team_choice = 1
            elif count_team_2 > count_team_1:
                team_choice = 2
            else:
                # Tie or all votes were 0 (unknown) - default to 2 
                # Note: We rely on the final per-frame mapping to apply the default '2'
                continue 

            stable_map[int(pid)] = int(team_choice)
        
        print(f"[INFO] TeamAssigner: Stable map generated for {len(stable_map)} players.")

        # build per-frame mapping using stable map
        final_per_frame = []
        for fi in range(n_frames):
            frame_map = {}
            # Ensure player_tracks has a valid list/dict for the frame index
            if fi < len(player_tracks) and player_tracks[fi]:
                for pid_raw in (player_tracks[fi] or {}).keys():
                    try:
                        pid = int(pid_raw)
                    except Exception:
                        pid = pid_raw
                    # Use stable map, default to Team 2 if player ID is new or missing
                    frame_map[pid] = stable_map.get(pid, 2)
            final_per_frame.append(frame_map)

        # save stable map to stub for future runs
        if stub_path:
            try:
                # Save the stable_map dict, not the per-frame list
                save_stub(stub_path, stable_map)
                print(f"[DBG] TeamAssigner: saved stable_map stub to {stub_path}")
            except Exception as e:
                print(f"[DBG] TeamAssigner: failed to save stub {stub_path}: {e}")

        return final_per_frame