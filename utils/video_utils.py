"""
utils/video_utils.py
Robust video read/write helpers with defensive checks and debug dumps.
"""

import cv2
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np

def read_video(video_path: str, reencode_if_needed: bool = True) -> Tuple[List[np.ndarray], Dict]:
    """
    Read video frames into memory. Returns (frames_list, metadata).
    metadata: {'fps': float, 'width': int, 'height': int, 'frame_count': int}
    If OpenCV fails to open, and reencode_if_needed is True, user should re-encode the file (ffmpeg).
    """
    meta = {'fps': 0.0, 'width': 0, 'height': 0, 'frame_count': 0}
    frames: List[np.ndarray] = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # Try simple fallback: let caller re-encode externally. Return empty with meta 0.
        print(f"[WARN] read_video: cv2 cannot open video: {video_path}")
        return frames, meta

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    meta.update({'fps': fps, 'width': width, 'height': height, 'frame_count': frame_count})
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Defensive: ensure frame is valid ndarray and 3-channel
        if frame is None:
            print(f"[WARN] read_video: got None frame at index {i}")
            i += 1
            continue
        if len(frame.shape) == 2:
            # convert grayscale to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frames.append(frame)
        i += 1
    cap.release()
    # final meta update
    meta['frame_count'] = len(frames)
    print(f"[DBG] read_video: read {len(frames)} frames, fps={meta['fps']}, size={meta['width']}x{meta['height']}")
    return frames, meta


def save_video(frames: List, output_video_path: str, fps: Optional[float] = 24.0, fallback_codecs: Optional[list] = None) -> bool:
    """
    Save frames to disk. Returns True on success, False otherwise.
    - Ensures parent dir exists.
    - Validates frames list.
    - Tries codecs in fallback_codecs if first fails.
    """
    if not frames:
        print(f"[WARN] save_video: no frames to save to {output_video_path}")
        return False

    Path(os.path.dirname(output_video_path) or '.').mkdir(parents=True, exist_ok=True)

    # Determine frame size from first frame
    try:
        h, w = frames[0].shape[:2]
    except Exception as e:
        print(f"[ERROR] save_video: invalid first frame: {e}")
        return False

    fps = fps or 24.0
    # codec order: try XVID then MJPG then MP4V
    codecs = fallback_codecs or ['XVID', 'MJPG', 'mp4v', 'H264']
    written = False
    for code in codecs:
        fourcc = cv2.VideoWriter_fourcc(*code)
        try:
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
            if not out.isOpened():
                print(f"[WARN] save_video: VideoWriter not opened with codec {code}")
                try:
                    out.release()
                except Exception:
                    pass
                continue
            for idx, frame in enumerate(frames):
                # Some frames might be grayscale; convert to BGR 3ch
                if frame is None:
                    print(f"[WARN] save_video: skipping None frame at index {idx}")
                    continue
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                # ensure correct size
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, (w, h))
                out.write(frame)
            out.release()
            # quick validation: file exists and size > small threshold
            if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 100:
                print(f"[DBG save_video] Initialized VideoWriter codec={code} path={output_video_path} fps={fps} size=({w}, {h})")
                print(f"[DBG save_video] done. wrote {len(frames)} frames to {output_video_path}")
                written = True
                break
        except Exception as e:
            print(f"[WARN] save_video: codec {code} failed with exception: {e}")
            try:
                out.release()
            except Exception:
                pass
    if not written:
        print(f"[ERROR] save_video: failed to write video to {output_video_path} with codecs {codecs}")
    return written


def dump_frame(frame, out_path):
    try:
        Path(os.path.dirname(out_path) or '.').mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path, frame)
        print(f"[DBG] dump_frame saved {out_path}")
    except Exception as e:
        print(f"[WARN] dump_frame failed: {e}")