#!/usr/bin/env python3
"""
main.py - robust runner for basketball_analysis (updated with pose, shot detection, metrics)

This version is defensive: optional modules (pose/activity/shot) are loaded if available.
If they're missing the pipeline will continue and produce output video/stubs for the rest of
the pipeline.
"""
import os
import sys
import traceback
import argparse
from typing import Tuple, Optional, List, Dict, Any

# ensure repo root import path (adjust if your layout differs)
folder = os.path.dirname(__file__)
sys.path.append(folder)

# Import utilities and modules from the repo (these must exist)
from utils.video_utils import read_video, save_video  # expects (frames, meta)
from utils import read_stub, save_stub  # generic stub helpers
from trackers import PlayerTracker, BallTracker
from team_assigner import TeamAssigner
from court_keypoint_detector import CourtKeypointDetector
from ball_aquisition import BallAquisitionDetector
from pass_and_interception_detector import PassAndInterceptionDetector
from tactical_view_converter import TacticalViewConverter
from speed_and_distance_calculator import SpeedAndDistanceCalculator
from drawers import (
    PlayerTracksDrawer,
    BallTracksDrawer,
    CourtKeypointDrawer,
    TeamBallControlDrawer,
    FrameNumberDrawer,
    PassInterceptionDrawer,
    TacticalViewDrawer,
    SpeedAndDistanceDrawer
)
from configs import (
    STUBS_DEFAULT_PATH,
    PLAYER_DETECTOR_PATH,
    BALL_DETECTOR_PATH,
    COURT_KEYPOINT_DETECTOR_PATH,
    OUTPUT_VIDEO_PATH
)

# -----------------------------
# Optional/extra modules (load if available)
# -----------------------------
_HAS_POSE_MODULES = False
_HAS_ACTIVITY_MODULE = False
_HAS_SHOT_MODULE = False

try:
    from pose_estimator import PoseEstimator
    _HAS_POSE_MODULES = True
except Exception as e:
    print(f"[WARN] pose_estimator not available: {e}. Pose steps will be skipped.")

try:
    from activity_classifier import ActivityClassifier
    _HAS_ACTIVITY_MODULE = True
except Exception as e:
    print(f"[WARN] activity_classifier not available: {e}. Activity classification will be skipped.")

try:
    from shot_detector import detect_shots_heuristic, evaluate_shots
    _HAS_SHOT_MODULE = True
except Exception as e:
    print(f"[WARN] shot_detector not available: {e}. Shot detection will be skipped.")
    # ensure names exist to avoid NameError later
    detect_shots_heuristic = None
    evaluate_shots = None

# jersey OCR utils (you said you've added)
try:
    from utils.jersey_ocr import batch_recognize_over_video, aggregate_player_numbers
except Exception:
    batch_recognize_over_video = None
    aggregate_player_numbers = None

# optional metrics helpers (defensive)
try:
    from utils.metrics import classification_metrics, mean_average_precision
except Exception:
    classification_metrics = None
    mean_average_precision = None

# defensive import for cv2 / numpy usage
try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

# -----------------------------
# helpers
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Basketball analysis runner (robust)")
    p.add_argument("input_video", type=str, help="Path to input video")
    p.add_argument("--output_video", type=str, default=OUTPUT_VIDEO_PATH, help="Path to write output video")
    p.add_argument("--stub_path", type=str, default=STUBS_DEFAULT_PATH, help="Directory for stubs")
    p.add_argument("--use_stubs", action="store_true", help="Use existing stubs instead of running detectors")
    p.add_argument("--force_rerun_detectors", action="store_true", help="Force rerun detectors even if stubs exist")
    p.add_argument("--fps", type=float, default=None, help="Force FPS when saving output (else use input video's fps or 24)")
    return p.parse_args()

def safe_call(name, fn, *args, **kwargs):
    """Call fn and capture exceptions; returns (ok, result)."""
    try:
        res = fn(*args, **kwargs)
        return True, res
    except Exception as e:
        print(f"[EXCEPTION] {name} failed: {e}")
        traceback.print_exc()
        return False, None

def fallback_opencv_write(frames, out_path, fps=24.0):
    """Simple fallback that writes frames with OpenCV VideoWriter (best-effort)."""
    if cv2 is None:
        print("[ERROR] fallback_opencv_write: cv2 not available")
        return False
    if not frames:
        print("[WARN] fallback_opencv_write: no frames to write")
        return False
    PathDir = os.path.dirname(out_path) or "."
    os.makedirs(PathDir, exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not out.isOpened():
        print("[ERROR] fallback_opencv_write: VideoWriter could not open")
        return False
    for f in frames:
        if f is None:
            continue
        if len(f.shape) == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        if f.shape[0] != h or f.shape[1] != w:
            f = cv2.resize(f, (w, h))
        out.write(f)
    out.release()
    ok = os.path.exists(out_path) and os.path.getsize(out_path) > 100
    if ok:
        print(f"[INFO] fallback_opencv_write: wrote {len(frames)} frames to {out_path}")
    else:
        print("[ERROR] fallback_opencv_write: output file invalid after write")
    return ok

def ensure_list_length(lst, length):
    """If lst is None or not length, convert to list of None of required length."""
    if isinstance(lst, list) and len(lst) == length:
        return lst
    return [None] * length

# -----------------------------
# main
# -----------------------------
def main():
    args = parse_args()
    print(f"[INFO] input={args.input_video} output={args.output_video} stubs={args.stub_path} use_stubs={args.use_stubs} force_rerun={args.force_rerun_detectors}")

    # 1) Read video frames + meta
    ok, read_out = safe_call("read_video", read_video, args.input_video)
    if not ok:
        print("[ERROR] read_video crashed -> exiting")
        sys.exit(1)

    try:
        frames, meta = read_out
    except Exception:
        print("[ERROR] read_video returned unexpected value. Exiting.")
        sys.exit(1)

    if not frames or len(frames) == 0:
        print("[ERROR] No frames read from input video. Exiting.")
        sys.exit(1)

    n_frames = len(frames)
    fps_input = meta.get("fps") or 30.0
    print(f"[DBG] Read {n_frames} frames; fps={fps_input} size={meta.get('width')}x{meta.get('height')}")

    # Determine read_from_stub policy
    read_from_stub = False
    if args.use_stubs and not args.force_rerun_detectors:
        read_from_stub = True

    # Create stub directory if not exists
    os.makedirs(args.stub_path, exist_ok=True)
    def stub(pname): return os.path.join(args.stub_path, pname)

    # 2) Initialize detectors/trackers
    print("[INFO] Initializing models/trackers/detectors...")
    player_tracker = PlayerTracker(PLAYER_DETECTOR_PATH)
    ball_tracker = BallTracker(BALL_DETECTOR_PATH)
    court_detector = CourtKeypointDetector(COURT_KEYPOINT_DETECTOR_PATH)
    team_assigner = TeamAssigner()
    ball_aq_detector = BallAquisitionDetector()
    pass_intercept_detector = PassAndInterceptionDetector()
    tactical_converter = TacticalViewConverter(court_image_path="./images/basketball_court.png")
    speed_calculator = SpeedAndDistanceCalculator(
        tactical_converter.width,
        tactical_converter.height,
        tactical_converter.actual_width_in_meters,
        tactical_converter.actual_height_in_meters,
        scale_factor=1.0
    )

    # 3) Run detectors / load stubs (players, ball, court)
    ok, player_tracks = safe_call("player_tracker.get_object_tracks", player_tracker.get_object_tracks, frames, read_from_stub=read_from_stub, stub_path=stub("player_track_stubs.pkl"))
    if not ok or player_tracks is None:
        print("[WARN] player_tracks missing -> using empty per-frame placeholders")
        player_tracks = [None] * n_frames
    else:
        # fix lengths
        if isinstance(player_tracks, list) and len(player_tracks) != n_frames:
            if len(player_tracks) < n_frames:
                player_tracks = player_tracks + [None] * (n_frames - len(player_tracks))
            else:
                player_tracks = player_tracks[:n_frames]
        try:
            save_stub(stub("player_track_stubs.pkl"), player_tracks)
        except Exception:
            pass

    ok, ball_tracks = safe_call("ball_tracker.get_object_tracks", ball_tracker.get_object_tracks, frames, read_from_stub=read_from_stub, stub_path=stub("ball_track_stubs.pkl"))
    if not ok or ball_tracks is None:
        print("[WARN] ball_tracks missing -> using empty per-frame placeholders")
        ball_tracks = [None] * n_frames
    else:
        if isinstance(ball_tracks, list) and len(ball_tracks) != n_frames:
            if len(ball_tracks) < n_frames:
                ball_tracks = ball_tracks + [None] * (n_frames - len(ball_tracks))
            else:
                ball_tracks = ball_tracks[:n_frames]
        try:
            save_stub(stub("ball_track_stubs.pkl"), ball_tracks)
        except Exception:
            pass

    ok, court_keypoints = safe_call("court_detector.get_court_keypoints", court_detector.get_court_keypoints, frames, read_from_stub=read_from_stub, stub_path=stub("court_key_points_stub.pkl"))
    if not ok or court_keypoints is None:
        print("[WARN] court_keypoints missing -> using empty per-frame placeholders")
        court_keypoints = [None] * n_frames
    else:
        if isinstance(court_keypoints, list) and len(court_keypoints) != n_frames:
            if len(court_keypoints) < n_frames:
                court_keypoints = court_keypoints + [None] * (n_frames - len(court_keypoints))
            else:
                court_keypoints = court_keypoints[:n_frames]
        try:
            save_stub(stub("court_key_points_stub.pkl"), court_keypoints)
        except Exception:
            pass

    # 4) Post-process ball tracks safely (remove wrong, interpolate)
    ok, ball_tracks = safe_call("remove_wrong_detections", ball_tracker.remove_wrong_detections, ball_tracks)
    if not ok or ball_tracks is None:
        ball_tracks = [None] * n_frames
    else:
        if isinstance(ball_tracks, list) and len(ball_tracks) != n_frames:
            if len(ball_tracks) < n_frames:
                ball_tracks = ball_tracks + [None] * (n_frames - len(ball_tracks))
            else:
                ball_tracks = ball_tracks[:n_frames]
    ok, ball_tracks = safe_call("interpolate_ball_positions", ball_tracker.interpolate_ball_positions, ball_tracks)
    if not ok or ball_tracks is None:
        ball_tracks = [None] * n_frames
    try:
        save_stub(stub("ball_track_stubs.pkl"), ball_tracks)
    except Exception:
        pass

    # --- begin: jersey OCR integration (fast sampling) ----
    ocr_results: List[Dict] = [ {} for _ in range(n_frames) ]
    player_number_map: Dict[Any, Any] = {}
    if batch_recognize_over_video is not None:
        ocr_stub_path = stub("jersey_ocr_results_stub.pkl")
        try:
            ocr_results = read_stub(read_from_stub, ocr_stub_path)
        except Exception:
            ocr_results = None

        if ocr_results is None or len(ocr_results) != n_frames:
            SAMPLE_RATE = 5
            OCR_ENGINE = "pytesseract"
            print(f"[INFO] Running FAST jersey OCR sampling every {SAMPLE_RATE} frames using {OCR_ENGINE}...")

            def _filter_player_tracks_for_ocr(frames, player_tracks, min_width=48, min_height=80):
                filtered_tracks = []
                for i, tracks in enumerate(player_tracks):
                    out = {}
                    for pid, pdata in (tracks or {}).items():
                        bbox = pdata.get("bbox") if isinstance(pdata, dict) else None
                        if not bbox:
                            continue
                        try:
                            x1,y1,x2,y2 = [int(float(v)) for v in bbox[:4]]
                            w = x2 - x1
                            h = y2 - y1
                        except Exception:
                            continue
                        if w >= min_width and h >= min_height:
                            out[pid] = pdata
                    filtered_tracks.append(out)
                return filtered_tracks

            filtered_tracks = _filter_player_tracks_for_ocr(frames, player_tracks)
            sample_indices = list(range(0, n_frames, SAMPLE_RATE))
            sampled_frames = [frames[i] for i in sample_indices]
            sampled_tracks = [filtered_tracks[i] for i in sample_indices]

            try:
                sampled_results = batch_recognize_over_video(sampled_frames, sampled_tracks, ocr_engine=OCR_ENGINE)
                ocr_results = [ {} for _ in range(n_frames) ]
                for idx, frame_idx in enumerate(sample_indices):
                    ocr_results[frame_idx] = sampled_results[idx]
                # propagate to neighbors
                for idx in sample_indices:
                    if idx-1 >= 0 and not ocr_results[idx-1]:
                        ocr_results[idx-1] = ocr_results[idx]
                    if idx+1 < n_frames and not ocr_results[idx+1]:
                        ocr_results[idx+1] = ocr_results[idx]
                try:
                    save_stub(ocr_stub_path, ocr_results)
                    print(f"[DBG] saved jersey OCR stub to {ocr_stub_path}")
                except Exception as e:
                    print(f"[WARN] saving jersey OCR stub failed: {e}")
            except Exception as e:
                print("[WARN] sampled jersey OCR failed:", e)
                ocr_results = [ {} for _ in range(n_frames) ]

        ocr_results = ensure_list_length(ocr_results, n_frames)
        if aggregate_player_numbers is not None and ocr_results:
            try:
                player_number_map = aggregate_player_numbers(ocr_results, min_confidence=0.4, min_occurrence=2)
                print("[INFO] Aggregated player numbers:", player_number_map)
            except Exception as e:
                print("[WARN] aggregate_player_numbers failed:", e)
                player_number_map = {}
            # Debug sample print
            print("[DBG OCR] Sample per-frame OCR results (first 3 frames):")
            for i in range(min(3, len(ocr_results))):
                sample_keys = list(ocr_results[i].keys())[:6]
                sample_dict = {k: ocr_results[i][k] for k in sample_keys}
                print(f" frame {i}: {sample_dict}")
    else:
        ocr_results = [ {} for _ in range(n_frames) ]
        player_number_map = {}

    # 5) Player team assignment
    ok, player_assignment = safe_call("team_assigner.get_player_teams_across_frames", team_assigner.get_player_teams_across_frames, frames, player_tracks, read_from_stub=read_from_stub, stub_path=stub("player_assignment_stub.pkl"))
    if not ok or player_assignment is None:
        print("[WARN] player_assignment missing -> using empty placeholders")
        player_assignment = [None] * n_frames
    else:
        if isinstance(player_assignment, list) and len(player_assignment) != n_frames:
            if len(player_assignment) < n_frames:
                player_assignment = player_assignment + [None] * (n_frames - len(player_assignment))
            else:
                player_assignment = player_assignment[:n_frames]
        try:
            save_stub(stub("player_assignment_stub.pkl"), player_assignment)
        except Exception:
            pass

    # 6) Ball acquisition
    ok, ball_acquisition = safe_call("ball_aq_detector.detect_ball_possession", ball_aq_detector.detect_ball_possession, player_tracks, ball_tracks)
    if not ok or ball_acquisition is None:
        print("[WARN] ball_acquisition missing -> fill with -1")
        ball_acquisition = [-1] * n_frames
    else:
        if len(ball_acquisition) != n_frames:
            if len(ball_acquisition) < n_frames:
                ball_acquisition = ball_acquisition + [-1] * (n_frames - len(ball_acquisition))
            else:
                ball_acquisition = ball_acquisition[:n_frames]
        try:
            save_stub(stub("ball_acquisition_stub.pkl"), ball_acquisition)
        except Exception:
            pass

    # 7) Passes and interceptions
    ok, passes = safe_call("pass_intercept_detector.detect_passes", pass_intercept_detector.detect_passes, ball_acquisition, player_assignment)
    if not ok or passes is None:
        passes = [-1] * n_frames
    else:
        if len(passes) != n_frames:
            if len(passes) < n_frames:
                passes = passes + [-1] * (n_frames - len(passes))
            else:
                passes = passes[:n_frames]
    ok, interceptions = safe_call("pass_intercept_detector.detect_interceptions", pass_intercept_detector.detect_interceptions, ball_acquisition, player_assignment)
    if not ok or interceptions is None:
        interceptions = [-1] * n_frames
    else:
        if len(interceptions) != n_frames:
            if len(interceptions) < n_frames:
                interceptions = interceptions + [-1] * (n_frames - len(interceptions))
            else:
                interceptions = interceptions[:n_frames]

    try:
        save_stub(stub("passes_stub.pkl"), passes)
        save_stub(stub("interceptions_stub.pkl"), interceptions)
    except Exception:
        pass

    # 8) Tactical view validate + transform
    ok, validated_keypoints = safe_call("tactical_converter.validate_keypoints", tactical_converter.validate_keypoints, court_keypoints)
    if not ok or validated_keypoints is None:
        validated_keypoints = [None] * n_frames

    ok, tactical_player_positions = safe_call("tactical_converter.transform_players_to_tactical_view", tactical_converter.transform_players_to_tactical_view, validated_keypoints, player_tracks)
    if not ok or tactical_player_positions is None:
        tactical_player_positions = [None] * n_frames
    else:
        if isinstance(tactical_player_positions, list) and len(tactical_player_positions) != n_frames:
            if len(tactical_player_positions) < n_frames:
                tactical_player_positions = tactical_player_positions + [None] * (n_frames - len(tactical_player_positions))
            else:
                tactical_player_positions = tactical_player_positions[:n_frames]

    # Debug prints
    print("\n--- DEBUG: Sample Tactical Player Positions and BBoxes ---")
    for i in range(min(3, len(tactical_player_positions))):
        print(f"[DEBUG tactical] frame={i} tactical={tactical_player_positions[i]}")
    if player_tracks and player_tracks[0]:
        try:
            first_pid = list(player_tracks[0].keys())[0]
            bbox = player_tracks[0][first_pid].get('bbox')
            print(f"[DEBUG player_tracks] frame=0, Player {first_pid} bbox: {bbox}")
        except Exception:
            pass
    print("----------------------------------------------------------\n")

    # SANITY: meters per pixel
    print("[SANITY] tactical dims:", tactical_converter.width, tactical_converter.height,
          "actual_m:", tactical_converter.actual_width_in_meters, tactical_converter.actual_height_in_meters)
    m_per_px = (tactical_converter.actual_width_in_meters / max(1, tactical_converter.width) + tactical_converter.actual_height_in_meters / max(1, tactical_converter.height)) / 2.0
    print(f"[SANITY] meters_per_pixel ~ {m_per_px:.6f} m/px")

    try:
        save_stub(stub("tactical_player_positions_stub.pkl"), tactical_player_positions)
        save_stub(stub("validated_keypoints_stub.pkl"), validated_keypoints)
    except Exception:
        pass

    # 9) Pose estimation (MediaPipe) - cached
    pose_results = None
    poses_stub_path = stub("poses_stub.pkl")
    if read_from_stub:
        try:
            pose_results = read_stub(True, poses_stub_path)
        except Exception:
            pose_results = None

    if pose_results is None and _HAS_POSE_MODULES:
        print("[INFO] Running MediaPipe pose over frames (this can take time)...")
        try:
            pose_est = PoseEstimator(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
            ok, pose_results = safe_call("pose_estimator.process_frames", pose_est.process_frames, frames)
            if not ok or pose_results is None:
                pose_results = [None] * n_frames
        except Exception as e:
            print("[WARN] PoseEstimator processing failed:", e)
            pose_results = [None] * n_frames
    elif pose_results is None:
        # no pose module, create empty stub list
        pose_results = [None] * n_frames

    pose_results = ensure_list_length(pose_results, n_frames)
    print("[DBG] Pose estimation sample:", pose_results[0] if pose_results else None)

    # 10) Activity classification
    activity = None
    activity_stub_path = stub("activity_stub.pkl")
    if read_from_stub:
        try:
            activity = read_stub(True, activity_stub_path)
        except Exception:
            activity = None

    if activity is None and _HAS_ACTIVITY_MODULE:
        try:
            act_cls = ActivityClassifier(meters_per_pixel=m_per_px, fps=fps_input, running_threshold_m_s=3.0)
            ok, activity = safe_call("activity_classifier.classify", act_cls.classify, tactical_player_positions)
            if not ok or activity is None:
                activity = [ {} for _ in range(n_frames) ]
        except Exception as e:
            print("[WARN] Activity classification failed:", e)
            activity = [ {} for _ in range(n_frames) ]
    elif activity is None:
        activity = [ {} for _ in range(n_frames) ]

    activity = ensure_list_length(activity, n_frames)
    print("[DBG] Activity sample (frame 0):", activity[0])

    # 11) Shot detection (heuristic)
    shots = None
    shots_stub_path = stub("shots_stub.pkl")
    if read_from_stub:
        try:
            shots = read_stub(True, shots_stub_path)
        except Exception:
            shots = None

    if shots is None and _HAS_SHOT_MODULE and detect_shots_heuristic is not None:
        try:
            shots = detect_shots_heuristic(ball_tracks=ball_tracks,
                                           tactical_ball_positions=None,
                                           player_positions=tactical_player_positions,
                                           pose_landmarks=pose_results,
                                           meters_per_pixel=m_per_px,
                                           fps=fps_input,
                                           upward_velocity_threshold_m_s=2.0,
                                           wrist_distance_px_threshold=80.0)
            shots = ensure_list_length(shots, n_frames)
        except Exception as e:
            print("[WARN] shot detection failed:", e)
            traceback.print_exc()
            shots = [False] * n_frames
    else:
        # fallback empty list
        shots = ensure_list_length(shots if shots is not None else [False]*n_frames, n_frames)

    try:
        save_stub(poses_stub_path, pose_results)
        save_stub(activity_stub_path, activity)
        save_stub(shots_stub_path, shots)
    except Exception:
        pass

    # 12) Speed & distance
    ok, player_distances_per_frame = safe_call("speed_calculator.calculate_distance", speed_calculator.calculate_distance, tactical_player_positions)
    if not ok or player_distances_per_frame is None:
        player_distances_per_frame = [dict() for _ in range(n_frames)]

    ok, player_speed_per_frame = safe_call("speed_calculator.calculate_speed", speed_calculator.calculate_speed, player_distances_per_frame, fps_input or 30)
    if not ok or player_speed_per_frame is None:
        player_speed_per_frame = [dict() for _ in range(n_frames)]

    try:
        save_stub(stub("player_distances_stub.pkl"), player_distances_per_frame)
        save_stub(stub("player_speeds_stub.pkl"), player_speed_per_frame)
    except Exception:
        pass

    # 13) Drawing pipeline
    print("[INFO] Running drawing pipeline...")
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    court_keypoint_drawer = CourtKeypointDrawer()
    team_ball_control_drawer = TeamBallControlDrawer()
    frame_number_drawer = FrameNumberDrawer()
    pass_and_interceptions_drawer = PassInterceptionDrawer()
    tactical_view_drawer = TacticalViewDrawer()
    speed_and_distance_drawer = SpeedAndDistanceDrawer()

    output_frames = list(frames)

    drawer_sequence = [
        ("PlayerTracksDrawer", player_tracks_drawer.draw, [player_tracks, player_assignment, ball_acquisition, ocr_results, player_number_map]),
        ("BallTracksDrawer", ball_tracks_drawer.draw, [ball_tracks]),
        ("CourtKeypointDrawer", court_keypoint_drawer.draw, [court_keypoints]),
        ("FrameNumberDrawer", frame_number_drawer.draw, []),
        ("TeamBallControlDrawer", team_ball_control_drawer.draw, [player_assignment, ball_acquisition]),
        ("PassInterceptionDrawer", pass_and_interceptions_drawer.draw, [passes, interceptions]),
        ("SpeedAndDistanceDrawer", speed_and_distance_drawer.draw, [player_tracks, player_distances_per_frame, player_speed_per_frame]),
        ("TacticalViewDrawer", tactical_view_drawer.draw, [tactical_converter.court_image_path,
                                                           tactical_converter.width,
                                                           tactical_converter.height,
                                                           tactical_converter.key_points,
                                                           tactical_player_positions,
                                                           player_assignment,
                                                           ball_acquisition])
    ]

    for name, fn, extra_args in drawer_sequence:
        try:
            print(f"[INFO] Running drawer: {name}")
            if extra_args:
                new_frames = fn(output_frames, *extra_args)
            else:
                new_frames = fn(output_frames)
            if new_frames is None:
                print(f"[WARN] {name}.draw returned None -> skipping and keeping previous frames")
                continue
            if not isinstance(new_frames, list) or len(new_frames) != len(output_frames):
                print(f"[WARN] {name}.draw returned invalid length ({None if new_frames is None else len(new_frames)}) - expected {len(output_frames)}. Skipping.")
                continue
            output_frames = new_frames
        except Exception as e:
            print(f"[EXCEPTION] drawer {name} crashed: {e}")
            traceback.print_exc()
            # continue with previous frames

    if not isinstance(output_frames, list) or len(output_frames) == 0:
        print("[WARN] Final frames invalid -> falling back to original frames")
        output_frames = frames

    # overlay pose/activity/shot markers (simple overlay so you can see them in the video)
    if cv2 is not None:
        for i in range(n_frames):
            frame = output_frames[i]
            # Draw simple pose landmarks if present (frame coords assumed to be in pixel space inside pose_results)
            pr = pose_results[i] if i < len(pose_results) else None
            if pr and isinstance(pr, dict):
                try:
                    for _, lm in (pr or {}).items():
                        # lm expected (x_px, y_px, visibility)
                        if not lm: 
                            continue
                        x, y, v = lm[:3]
                        if v is None: 
                            continue
                        try:
                            cv2.circle(frame, (int(round(x)), int(round(y))), 2, (0,255,0), -1)
                        except Exception:
                            pass
                except Exception:
                    pass
            # draw activity labels and shot text
            try:
                # activity is dict per frame mapping player_id -> label
                acts = activity[i] if i < len(activity) else {}
                p_tracks = player_tracks[i] if i < len(player_tracks) else {}
                for pid, label in (acts or {}).items():
                    try:
                        pdata = p_tracks.get(pid) if isinstance(p_tracks, dict) else None
                        if not pdata:
                            continue
                        bbox = pdata.get("bbox")
                        if bbox and len(bbox) >= 4:
                            x1,y1 = int(float(bbox[0])), int(float(bbox[1]))
                            text = str(label)
                            jn = player_number_map.get(pid)
                            if jn:
                                text = f"#{jn} {text}"
                            cv2.putText(frame, text, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
                    except Exception:
                        continue
                if shots[i]:
                    cv2.putText(frame, "SHOT", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3, cv2.LINE_AA)
            except Exception:
                pass
            output_frames[i] = frame

    # 14) Save video
    out_fps = args.fps or fps_input or 24.0
    print(f"[INFO] Saving output video to {args.output_video} (fps={out_fps})")
    ok = save_video(output_frames, args.output_video, fps=out_fps)
    if not ok:
        print("[WARN] primary save_video failed, trying fallback_opencv_write")
        try:
            ok = fallback_opencv_write(output_frames, args.output_video, fps=out_fps)
        except Exception:
            ok = False
    if not ok:
        print("[ERROR] All attempts to save video failed. Check permissions and codecs.")
        sys.exit(2)

    print("[INFO] Pipeline complete. Output written to:", args.output_video)

    # 15) Metrics - only if GT files are present in stub folder
    # Shots metrics: look for shots_gt.pkl (list of booleans or frame indices)
    shots_gt = None
    try:
        shots_gt = read_stub(True, stub("shots_gt.pkl"))
    except Exception:
        shots_gt = None

    if shots_gt is not None and _HAS_SHOT_MODULE and evaluate_shots is not None:
        try:
            # normalize gt to boolean list
            if isinstance(shots_gt, list) and all(isinstance(x, (bool,int)) for x in shots_gt):
                if all(isinstance(x, bool) for x in shots_gt) and len(shots_gt) == len(shots):
                    gt_bool = shots_gt
                else:
                    # assume list of indices
                    gt_bool = [False] * n_frames
                    for idx in shots_gt:
                        if 0 <= int(idx) < n_frames:
                            gt_bool[int(idx)] = True
            else:
                gt_bool = [False] * n_frames
            metrics_shots = evaluate_shots(shots, gt_bool)
            print("[METRICS] Shot detection:", metrics_shots)
        except Exception as e:
            print("[WARN] evaluate_shots failed:", e)
    else:
        if shots_gt is None:
            print("[INFO] No shots_gt.pkl found -> skipping shot metrics.")
        else:
            print("[INFO] Shot evaluation module not available -> skipping shot metrics.")

    # Team / jersey metrics: only if stubs present (skipping heavy logic if not)
    team_gt = None
    try:
        team_gt = read_stub(True, stub("team_gt.pkl"))
    except Exception:
        team_gt = None

    if team_gt is not None:
        print("[INFO] team_gt.pkl found. You can implement evaluation logic here to compare player_assignment -> team_gt.")
    else:
        print("[INFO] No team_gt.pkl found -> skipping team assignment metrics.")

    jersey_gt = None
    try:
        jersey_gt = read_stub(True, stub("jersey_gt.pkl"))
    except Exception:
        jersey_gt = None

    if jersey_gt is not None:
        print("[INFO] jersey_gt.pkl found. You can implement comparison between player_number_map and jersey_gt here.")
    else:
        print("[INFO] No jersey_gt.pkl found -> skipping jersey OCR metrics.")

if __name__ == "__main__":
    main()