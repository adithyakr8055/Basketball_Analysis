# 🏀 Basketball Analytics System 
### Automated Player Tracking, Event Detection & Tactical Analytics using Computer Vision + Deep Learning

---

## 📌 Summary

**Basketball Analytics System ** is an end-to-end computer-vision pipeline that processes raw basketball match videos and automatically generates:

- 👤 Player detection & tracking  
- 🏀 Ball detection & tracking  
- 📐 Court keypoint detection + homography  
- 🔵🔴 Team assignment (CLIP + KMeans fallback)  
- 🔢 Jersey number OCR (EasyOCR + preprocessing)  
- 🧍 Pose estimation  
- 🔄 Pass & interception detection  
- 🎯 Shot detection & rebounds  
- ⚡ Speed & distance metrics  
- 🗺 Top-down tactical analytics  

The system is **modular** — missing modules are skipped safely.

---

# 📁 Repository Structure (Annotated)

```
basketball_analysis/
│
├── main.py                         # 🚀 Main pipeline orchestrator
├── requirements.txt                # 📦 Python dependencies
│
├── models/                         # 🤖 YOLO + keypoint detector models (Git LFS)
│     ├── player_detector.pt
│     ├── ball_detector_model.pt
│     └── court_keypoint_detector.pt
│
├── input_videos/                   # 🎥 Input videos
├── output_videos/                  # 🧵 Annotated processed videos
│
├── utils/                          # 🧰 Core utilities
│     ├── jersey_ocr.py             # 🔢 Jersey OCR pipeline
│     ├── ocr_smoothing.py          # 🧠 Number smoothing (temporal)
│     ├── video_utils.py            # 🎬 Read/save videos
│     ├── bbox_utils.py             # 🔲 BBox helpers, distance functions
│
├── trackers/                       # 🛰 DeepSORT + Kalman tracking
├── team_assigner/                  # 🔵🔴 CLIP + KMeans team assignment
├── court_keypoint_detector/        # 📐 Court line detection + homography
├── drawers/                        # ✏️ Drawing overlays
├── ball_aquisition/                # 🏀 Ball possession logic
│
├── tools/                          # 🔧 Debugging scripts (OCR tester, crops)
│
├── pass_and_interception_detector.py # 🔄 Pass/interception detection
├── tactical_view_converter/        # 🗺 Tactical top-down projection
├── speed_and_distance_calculator/  # ⚡ Speed/distance metrics
├── pose_estimator.py               # 🧍 Pose estimation (MediaPipe)
├── shot_detector.py                # 🎯 Shot detection logic
│
├── stubs/                          # 💾 Cached intermediate outputs (PKL files)
│     ├── player_tracks.pkl
│     ├── ball_tracks.pkl
│     ├── passes_stub.pkl
│     └── jersey_ocr_results.pkl
│
└── training_notebooks/             # 📓 Model training experiments
```

---

# ⚙️ Installation & Setup

## 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your_user>/Basketball_Analysis.git
cd Basketball_Analysis
```

---

## 2️⃣ Git LFS Setup (Required for .pt & video files)
```bash
git lfs install
git lfs pull
```

---

## 3️⃣ Create & Activate Virtual Environment (macOS/Linux)

```bash
python3 -m venv basketball_venv
source basketball_venv/bin/activate
```

Your environment name (as per your system):

```bash
source basketball_venv311/bin/activate
```

To deactivate:
```bash
deactivate
```

---

## 4️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Full Pipeline

Place your input video in:

```
input_videos/video1.mp4
```

## Basic run:
```bash
python main.py input_videos/video1.mp4 --output_video output_videos/output.avi
```

## Force rerun detectors (ignore cached stubs):
```bash
python main.py input_videos/video1.mp4 --output_video output_videos/output.avi --force_rerun_detectors
```

---

# 🧠 MODULE EXPLANATION 

## 👤 Player Detection & Tracking
- YOLO-based detector  
- DeepSORT assigns stable track IDs  
- Kalman filter smooths trajectory  

📦 Output: `player_tracks.pkl`

---

## 🏀 Ball Detection & Tracking
- YOLO small-object detection  
- Selects best ball bbox per frame  
- Interpolates missing frames  

📦 Output: `ball_tracks.pkl`

---

## 📐 Court Keypoint Detection & Homography
- Detects court intersections  
- Computes pixel → real-world transformation  
- Used for speed, distance, tactical view  

📦 Output: `court_keypoints.pkl`

---

## 🔵🔴 Team Assignment
### 1. CLIP classifier (preferred)  
Classifies jersey crops as:
- `"white shirt"`
- `"dark blue shirt"`

### 2. KMeans fallback  
Used when:
- transformers not installed  
- GPU unavailable  
- fast_mode=True  

### 3. Temporal Voting  
Ensures stable team assignment per player ID.

📦 Output: `team_assignment_stub.pkl`

---

## 🔢 Jersey Number OCR
### Pipeline:
1. Crop upper jersey region  
2. Apply 4 preprocessing variations:  
   - Bilateral + adaptive threshold  
   - CLAHE + Otsu threshold  
   - Sharpen + upsample  
   - Equalize histogram  
3. Deskew  
4. EasyOCR (preferred) or pytesseract fallback  
5. Temporal smoothing  
6. Confidence-based filtering  
7. Majority vote per-player  

📦 Output: `jersey_ocr_results.pkl`

---

## 🏀 Ball Possession Detection
Uses:
- Distance to player  
- Containment ratio (ball inside bbox)  
- Consecutive frames (≥11 frames threshold)  
- Backfilling for continuity  

📦 Output: `ball_acquisition.pkl`

---

## 🔄 Pass & Interception Detection
Rules:
- If possessor changes **within same team → PASS**  
- If possessor changes **between teams → INTERCEPTION**  

📦 Outputs:
- `passes_stub.pkl`
- `interceptions_stub.pkl`

---

## ⚡ Speed & Distance Metrics
Uses court homography → real meters.

Outputs:
- Instant speed  
- Top speed  
- Average speed  
- Total distance  

📦 Output: `player_speeds.pkl`

---

## 🗺 Tactical Analytics (Top-Down Court)
- Converts positions to 2D court  
- Shows spacing, formations, player roles, shot maps, heatmaps  

---

# 🔧 Debug Scripts

Run OCR tester:
```bash
python tools/ocr_full_test_player.py
```

Fix imports:
```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
```

---

# 🌐 Git + LFS Push Instructions

## Check remote:
```bash
git remote -v
```

## Set remote:
```bash
git remote set-url origin https://github.com/adithyakr8055/Basketball_Analysis.git
```

## Track large files:
```bash
git lfs track "*.pt"
git lfs track "*.mp4"
git lfs track "*.avi"
git add .gitattributes
```

## Stage all:
```bash
git add .
```

## Commit:
```bash
git commit -m "Full BVA project"
```

## Push:
```bash
git push -u origin main
```

---

# 📜 License
MIT License 
