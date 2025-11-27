"""
TeamAssigner
Assign players to teams using either CLIP (preferred) or a fast k-means color heuristic (fallback).

Hybrid logic:
- If transformers + CLIP are installed and fast_mode=False (default):
    -> Use CLIP on the jersey region of each player.
- If CLIP is unavailable OR CLIP returns "unknown":
    -> Fall back to k-means + brightness on the jersey region.
- Results are stabilized across frames using majority vote.
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
sys.path.append(os.path.join(folder, "../"))  # repo root

try:
    from utils import read_stub, save_stub  # your existing utils (pickle stubs)
except ImportError:
    print("[ERROR] Could not import read_stub/save_stub from utils. Check your path setup.")
    def read_stub(*args, **kwargs): return None
    def save_stub(*args, **kwargs): pass


def _jersey_roi(image_crop: np.ndarray) -> np.ndarray:
    """
    Return only the approximate jersey region (torso) of a player crop.
    This makes color detection focus on the shirt instead of floor/shorts/background.
    """
    if image_crop is None or image_crop.size == 0:
        return image_crop

    h, w = image_crop.shape[:2]

    # If very small, upsample first for more stable k-means / CLIP
    if h * w < 100:
        image_crop = cv2.resize(image_crop, (64, 128), interpolation=cv2.INTER_LINEAR)
        h, w = image_crop.shape[:2]

    # Heuristic torso: from ~20% to 70% of height, 20% to 80% of width
    y1 = int(0.20 * h)
    y2 = int(0.70 * h)
    x1 = int(0.20 * w)
    x2 = int(0.80 * w)

    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))

    roi = image_crop[y1:y2, x1:x2]
    return roi if roi.size > 0 else image_crop


def _kmeans_color_label(image_crop: Optional[np.ndarray], k: int = 2) -> str:
    """
    Fallback labeling using k-means color clustering on the *jersey region* of the player crop.
    Returns either "white shirt" or "dark blue shirt" (to match default labels),
    or "unknown" if the crop is invalid.
    """
    if image_crop is None or image_crop.size == 0:
        return "unknown"
    try:
        # Ensure we have a 3-channel BGR image
        if len(image_crop.shape) == 2:
            image_crop = cv2.cvtColor(image_crop, cv2.COLOR_GRAY2BGR)

        jersey_roi = _jersey_roi(image_crop)
        data = jersey_roi.reshape((-1, 3)).astype(np.float32)

        # k-means on jersey colors
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, flags)
        counts = np.bincount(labels.flatten(), minlength=k)
        dominant = centers[np.argmax(counts)]  # BGR center of dominant jersey color

        b, g, r = map(float, dominant)
        # Perceived brightness
        brightness = 0.299 * r + 0.587 * g + 0.114 * b

        # Threshold can be tuned for your footage; 130 works well for white vs navy
        return "white shirt" if brightness > 130 else "dark blue shirt"
    except Exception:
        return "unknown"


class TeamAssigner:
    """
    Assign players to teams using CLIP (if available) or a k-means color heuristic.

    Args:
        team_1_class_name: textual label representing team 1 (default "white shirt")
        team_2_class_name: textual label representing team 2 (default "dark blue shirt")
        fast_mode: if True, skip CLIP entirely and always use k-means heuristic
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

    # ------------------ CLIP LOADING ------------------ #

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
            self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
            self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
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

    # ------------------ CROPPING ------------------ #

    def _crop_safe(
        self,
        frame: Optional[np.ndarray],
        bbox: Optional[List[float]],
        player_id: int = -1,
        frame_num: int = -1,
    ) -> Optional[np.ndarray]:
        """
        Safely crop bbox from frame. Returns None if invalid.
        Ensures bounding box clipping inside the frame and minimal crop size, with debug logging.
        """
        if frame is None or bbox is None:
            if player_id != -1 and frame_num != -1:
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
            ch, cw = crop.shape[:2]

            # If crop very small, resize to a reasonable size for processing
            if ch < 20 or cw < 20:
                if player_id != -1 and frame_num != -1:
                    print(
                        f"[DBG] TeamAssigner: crop too small ({crop.shape[0]}x{crop.shape[1]}) "
                        f"for player {player_id} frame {frame_num}"
                    )
                crop = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_LINEAR)

            return crop
        except Exception as e:
            if player_id != -1 and frame_num != -1:
                print(f"[DBG] TeamAssigner: _crop_safe exception for player {player_id} frame {frame_num}: {e}")
            return None

    # ------------------ COLOR / CLIP LABELING ------------------ #

    def get_player_color_clip(
        self,
        frame: np.ndarray,
        bbox: List[float],
        player_id: int = -1,
        frame_num: int = -1,
    ) -> str:
        """
        Use CLIP to classify the player's jersey between the two class names.
        Returns self.team_1_class_name or self.team_2_class_name or "unknown".
        """
        if not self._model_loaded:
            if not self.load_model():
                return "unknown"

        crop = self._crop_safe(frame, bbox, player_id, frame_num)
        if crop is None:
            return "unknown"

        jersey = _jersey_roi(crop)

        try:
            rgb = cv2.cvtColor(jersey, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)

            # Descriptive CLIP text prompts, but map back to simple labels
            text_prompts = [
                f"a basketball player wearing a {self.team_1_class_name}",
                f"a basketball player wearing a {self.team_2_class_name}",
            ]

            inputs = self.processor(text=text_prompts, images=pil, return_tensors="pt", padding=True)
            try:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            except Exception:
                pass

            outputs = self.model(**inputs)
            logits = outputs.logits_per_image  # shape (1, n_classes)
            probs = logits.softmax(dim=1)
            idx = int(probs.argmax(dim=1)[0])

            # Map back to canonical class names used in rest of code
            return self.team_1_class_name if idx == 0 else self.team_2_class_name
        except Exception as e:
            print(f"[DBG] TeamAssigner: CLIP classification error: {e}")
            return "unknown"

    def get_player_color(
        self,
        frame: np.ndarray,
        bbox: List[float],
        player_id: int = -1,
        frame_num: int = -1,
    ) -> str:
        """
        Public method that returns a textual class for the player's crop.
        Uses CLIP when available and not in fast_mode; otherwise k-means fallback.
        """
        crop = self._crop_safe(frame, bbox, player_id, frame_num)
        if crop is None:
            return "unknown"

        # fast_mode: always k-means
        if self.fast_mode:
            return _kmeans_color_label(crop)

        # try CLIP first
        label = self.get_player_color_clip(frame, bbox, player_id, frame_num)
        if not label or label == "unknown":
            label = _kmeans_color_label(crop)
        return label

    # ------------------ SINGLE-PLAYER TEAM (legacy) ------------------ #

    def get_player_team(self, frame: np.ndarray, player_bbox: List[float], player_id: int) -> int:
        """
        Return team id (1 or 2) for a single player, using cached mapping if possible.
        Defaults to team 2 if classification uncertain.

        NOTE: This method is primarily for compatibility; the majority-vote logic is preferred.
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        label = self.get_player_color(frame, player_bbox, player_id=player_id)
        team_id = 2
        if label == self.team_1_class_name:
            team_id = 1
        elif label == "unknown":
            crop = self._crop_safe(frame, player_bbox, player_id=player_id)
            if crop is not None:
                jersey = _jersey_roi(crop)
                b, g, r = cv2.mean(jersey)[:3]
                brightness = 0.299 * r + 0.587 * g + 0.114 * b
                if brightness > 130:
                    team_id = 1
        self.player_team_dict[player_id] = team_id
        return team_id

    # ------------------ MULTI-FRAME MAJORITY VOTE ------------------ #

    def get_player_teams_across_frames(
        self,
        video_frames: List[np.ndarray],
        player_tracks: List[Dict[int, Dict]],
        read_from_stub: bool = False,
        stub_path: Optional[str] = None,
        min_votes: int = 3,
    ) -> List[Dict[int, int]]:
        """
        For every player across frames, collect team votes (1 or 2), then:
        - Build a stable team_id for each player via majority vote.
        - Return per-frame mapping: player_id -> stable team_id.
        """
        # Try to load stable_map from stub
        stable_map: Dict[int, int] = {}
        if stub_path:
            try:
                st = read_stub(read_from_stub, stub_path)
                if st is not None and isinstance(st, dict):
                    stable_map = st
                    print(f"[DBG] TeamAssigner: loaded stable_map from stub with {len(stable_map)} players")
            except Exception:
                pass

        n_frames = len(video_frames)

        # If stable_map exists, apply it directly
        if stable_map and player_tracks:
            out: List[Dict[int, int]] = []
            for frame_tracks in player_tracks:
                frame_map: Dict[int, int] = {}
                for pid_raw in (frame_tracks or {}).keys():
                    try:
                        pid = int(pid_raw)
                    except Exception:
                        pid = pid_raw
                    frame_map[pid] = stable_map.get(pid, 2)
                out.append(frame_map)
            if len(out) == n_frames:
                return out
            else:
                print(
                    f"[WARN] TeamAssigner: Loaded stable map but player_tracks length mismatch "
                    f"({len(player_tracks)} vs {n_frames}). Recalculating."
                )

        # Ensure model loaded if available (CLIP)
        if not self.fast_mode and _HAS_CLIP:
            self.load_model()

        # Collect votes: pid -> list of team ids (1,2, or 0 for truly unknown)
        votes: Dict[int, List[int]] = defaultdict(list)

        for fi, frame_tracks in enumerate(player_tracks or []):
            if not frame_tracks:
                continue
            if fi >= n_frames:
                print(f"[WARN] Frame index {fi} out of bounds for video frames. Stopping vote collection.")
                break

            frame = video_frames[fi]

            for pid_raw, pdata in (frame_tracks or {}).items():
                try:
                    pid = int(pid_raw)
                except Exception:
                    pid = pid_raw
                bbox = pdata.get("bbox") if isinstance(pdata, dict) else None
                if bbox is None:
                    continue

                label = self.get_player_color(frame, bbox, pid, fi)

                if label == self.team_1_class_name:
                    tid = 1
                elif label == self.team_2_class_name:
                    tid = 2
                else:
                    crop = self._crop_safe(frame, bbox)
                    if crop is not None and crop.size > 0:
                        jersey = _jersey_roi(crop)
                        b, g, r = cv2.mean(jersey)[:3]
                        brightness = 0.299 * r + 0.587 * g + 0.114 * b
                        tid = 1 if brightness > 130 else 2
                    else:
                        tid = 0
                votes[pid].append(tid)

        # Build stable and fallback maps
        stable_map = {}
        fallback_map = {}

        for pid, arr in votes.items():
            valid_votes = [t for t in arr if t in (1, 2)]
            total_valid = len(valid_votes)

            if total_valid > 0:
                cnt = Counter(valid_votes)
                count_team_1 = cnt.get(1, 0)
                count_team_2 = cnt.get(2, 0)
                fb_team = 1 if count_team_1 >= count_team_2 else 2
                fallback_map[int(pid)] = fb_team

            if total_valid < min_votes:
                print(f"[DBG] TeamAssigner: Player {pid} skipped - only {total_valid} valid votes (min={min_votes})")
                continue

            cnt = Counter(valid_votes)
            count_team_1 = cnt.get(1, 0)
            count_team_2 = cnt.get(2, 0)

            if count_team_1 > count_team_2:
                team_choice = 1
            elif count_team_2 > count_team_1:
                team_choice = 2
            else:
                continue

            stable_map[int(pid)] = int(team_choice)

        print(f"[INFO] TeamAssigner: Stable map generated for {len(stable_map)} players.")

        # Build per-frame mapping
        final_per_frame: List[Dict[int, int]] = []
        for fi in range(n_frames):
            frame_map: Dict[int, int] = {}
            if fi < len(player_tracks) and player_tracks[fi]:
                for pid_raw in (player_tracks[fi] or {}).keys():
                    try:
                        pid = int(pid_raw)
                    except Exception:
                        pid = pid_raw

                    if pid in stable_map:
                        frame_map[pid] = stable_map[pid]
                    elif pid in fallback_map:
                        frame_map[pid] = fallback_map[pid]
                    else:
                        frame_map[pid] = 2
            final_per_frame.append(frame_map)

        # Save stable_map for future runs
        if stub_path:
            try:
                save_stub(stub_path, stable_map)
                print(f"[DBG] TeamAssigner: saved stable_map stub to {stub_path}")
            except Exception as e:
                print(f"[DBG] TeamAssigner: failed to save stub {stub_path}: {e}")

        return final_per_frame


def get_player_teams_across_frames(
    video_frames: List[np.ndarray],
    player_tracks: List[Dict[int, Dict]],
    read_from_stub: bool = False,
    stub_path: Optional[str] = None,
    fast_mode: bool = False,
    team_1_class_name: str = "white shirt",
    team_2_class_name: str = "dark blue shirt",
    min_votes: int = 3,
    device: Optional[str] = None,
) -> List[Dict[int, int]]:
    """
    Convenience wrapper if you want to call it as a simple function.
    """
    assigner = TeamAssigner(
        team_1_class_name=team_1_class_name,
        team_2_class_name=team_2_class_name,
        fast_mode=fast_mode,
        device=device,
    )
    return assigner.get_player_teams_across_frames(
        video_frames=video_frames,
        player_tracks=player_tracks,
        read_from_stub=read_from_stub,
        stub_path=stub_path,
        min_votes=min_votes,
    )
