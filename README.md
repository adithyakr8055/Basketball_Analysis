Basketball Analytics System (BVA)

Automated Player Tracking, Event Detection & Tactical Analytics Using Computer Vision + Deep Learning

⸻

Summary (elevator pitch)

BVA is an end-to-end computer-vision pipeline that processes basketball match video and produces:
	•	player tracking & IDs
	•	ball tracking
	•	court detection & homography
	•	team assignment & jersey OCR
	•	pose estimation
	•	speed/distance metrics
	•	passes, interceptions, shots, rebounds
	•	tactical analytics (heatmaps, shot maps, trajectories)

This repo is modular so missing optional modules will be skipped gracefully (see main.py).

⸻

Quick status

Repo root: basketball_analysis/ — run scripts from this root so imports like from utils import jersey_ocr work.

Large files (models, videos) should be managed by Git LFS.

⸻

Repo structure (annotated)

basketball_analysis/
├─ main.py                      # Main runner (orchestration, robust/fallback logic)
├─ requirements.txt             # Python dependencies
├─ models/                      # Large model files tracked via Git LFS (.pt, weights)
├─ input_videos/                # Place input videos here
├─ output_videos/               # Annotated/processed videos written here
├─ utils/                       # Core shared utilities
│  ├─ jersey_ocr.py             # OCR helper (preprocessing, easyocr/pytesseract wrapper)
│  ├─ video_utils.py            # read_video / save_video helpers
│  ├─ bbox_utils.py             # bbox helpers, measure_distance, centers, etc.
│  └─ ocr_smoothing.py          # temporal smoothing helper (create this if missing)
├─ trackers/                    # Player & ball trackers (DeepSORT + Kalman wrappers)
├─ court_keypoint_detector/     # Court line/keypoint detection + homography
├─ team_assigner/               # Team assignment (CLIP + kmeans fallback)
├─ drawers/                     # Drawing overlays for output frames
├─ tools/                       # Small helper/debugging scripts (distance_debugger, run_ocr_on_crops)
├─ stubs/                       # Pickled intermediate outputs (player_tracks, ball_tracks, etc.)
└─ training_notebooks/          # Notebooks used during experimentation/training


⸻

Installation & Setup

1) Clone & Git LFS

git clone https://github.com/<your_user>/Basketball_Analysis.git
cd Basketball_Analysis
git lfs install
git lfs pull

If you use SSH remote replace https://... with git@github.com:....

2) Create & activate venv (recommended)

macOS / Linux (using venv):

python3 -m venv basketball_venv
source basketball_venv/bin/activate
# when done: deactivate

If your venv is named basketball_venv311 (as your prompt shows):

source basketball_venv311/bin/activate

Conda users:

conda create -n bva python=3.10
conda activate bva

3) Install Python dependencies

pip install -r requirements.txt

If you need GPU PyTorch, install the correct wheel for your CUDA version (visit pytorch.org).

⸻

Running the pipeline

Basic run (from repo root):

python main.py input_videos/video1.mp4 --output_video output_videos/result.avi

Force rerun detectors and ignore cached stubs:

python main.py input_videos/video1.mp4 --output_video output_videos/result.avi --force_rerun_detectors

OCR debug script (examples)
	•	tools/run_ocr_on_crops.py — runs OCR on saved crop images and writes CSV output.
	•	tools/ocr_full_test_player.py — test OCR on a single player across frames.

Run from repo root (so imports resolve):

python tools/ocr_full_test_player.py

If you see ModuleNotFoundError: No module named 'utils' run:

export PYTHONPATH="$PWD:$PYTHONPATH"
python tools/ocr_full_test_player.py


⸻

Important implementation notes (for panel)

Main pipeline (main.py)
	•	Orchestrates reading frames, running detectors/trackers, computing homography, converting to tactical/top-down coordinates, running event detectors (ball acquisition, pass detection), applying OCR on sampled frames, smoothing OCR results, drawing overlays and saving final video.
	•	Is defensive: uses read_stub/save_stub to cache heavy outputs and gracefully skips optional modules (pose, shot) if dependencies absent.

Team assignment
	•	Attempts CLIP-based classification (if transformers + torch available) to label player crops like “white shirt” vs “dark blue shirt”.
	•	Falls back to a k-means color heuristic when CLIP is not available or when fast_mode=True.
	•	The final mapping is computed by voting across frames to produce a stable player_id -> team_id map.

Jersey OCR
	•	utils/jersey_ocr.py crops an upper region of player bbox, runs multiple preprocess methods (CLAHE, bilateral + adaptive threshold, sharpen+resize, histogram equalization), deskews, then calls EasyOCR (preferred) or pytesseract as fallback.
	•	main.py samples frames at SAMPLE_RATE (default 5) to speed up OCR, then propagates results to neighboring frames and performs temporal aggregation.
	•	If OCR quality is low, switch to easyocr and/or reduce SAMPLE_RATE so you process more frames during debugging.

Ball acquisition & passes
	•	Ball acquisition chooses the best ball bbox per frame and assigns a holding player using distance/containment heuristics and a consecutive-frame threshold before confirming possession.
	•	Pass detection uses change-of-possessor logic and checks team assignments to label passes vs interceptions.

Distances & homography
	•	Accurate player distances rely on a correct homography between image and court coordinates. If homography is missing or miscomputed, distances will appear huge or inconsistent.

⸻

Recommended quick fixes (based on issues you reported)
	•	Create utils/ocr_smoothing.py if missing — add a simple temporal smoothing function so main.py can import it.
	•	Ensure PYTHONPATH contains repo root so from utils import jersey_ocr works when running tools directly.
	•	Prefer easyocr in pipeline if installed — it handles digits on textured jerseys better.

⸻

Git + LFS push (safe sequence)
	1.	Ensure LFS tracking for large types and commit .gitattributes:

git lfs install
git lfs track "*.pt" "*.mp4" "*.avi"
git add .gitattributes

	2.	Stage & commit everything:

git add .
git commit -m "Project snapshot: BVA"

	3.	Set remote if needed and push:

git remote set-url origin https://github.com/adithyakr8055/Basketball_Analysis.git
git branch -M main
git push -u origin main

If you get remote origin already exists, use git remote -v to inspect and git remote set-url origin <url> to update.

⸻


License

Choose one (edit inside this README): MIT / Apache-2.0 / Proprietary (internal). Let me know which and I will add full license text.

⸻

If you want, I will create the missing utils/ocr_smoothing.py stub next and push both the README and stub commands to your terminal. Tell me which files you want me to create now.
