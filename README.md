🏀 Basketball Analytics System (BVA)

Automated Player Tracking, Event Detection & Tactical Analytics Using Computer Vision + Deep Learning

⸻

📌 Overview

The Basketball Analytics System (BVA) is a complete end-to-end computer-vision pipeline that processes basketball match videos and automatically extracts:
	•	Player detections
	•	Ball detection & tracking
	•	Court keypoint detection
	•	Pose estimation
	•	Player speed & distance
	•	Team classification
	•	Passes
	•	Interceptions
	•	Field goal attempts
	•	Rebounds
	•	Shot charts
	•	Tactical positioning
	•	Activity classification

Built using YOLO models, Keypoint Estimators, DeepSORT, Kalman Filters, KMeans, OCR, Pose Models, and custom logic.

This project supports real-time analysis and offline video processing.

⸻

🚀 Features

🔥 1. Player & Ball Tracking
	•	YOLO-based player detection
	•	YOLO-based ball detector
	•	DeepSORT & Kalman tracking
	•	Track ID stabilisation

🧍‍♂️🧍‍♀️ 2. Pose Estimation
	•	Key body landmarks
	•	Pose classification
	•	Player posture analysis

🎯 3. Court Keypoint Detection
	•	Full court line/keypoint detection
	•	Homography transform
	•	Per-player x,y metric locations

🔵🔴 4. Team Assignment
	•	Jersey color clustering
	•	KMeans color grouping
	•	OCR-based jersey number extraction

🏃‍♂️ 5. Speed & Distance Metrics
	•	m/s and km/h speed calculation
	•	Cumulative distance tracking
	•	Heatmaps & trajectories

🔀 6. Event Detection
	•	Passes
	•	Interceptions
	•	Defensive/offensive movement
	•	Shooting actions
	•	Shot success / failure
	•	Rebounds

📊 7. Tactical Analytics
	•	Player spacing
	•	Team formation
	•	Off-ball movement
	•	Shot map
  
📁 Project Structure
basketball_analysis/
│
├── main.py                           # Main pipeline runner
│
├── models/                           # LFS-tracked model binaries
│   ├── ball_detector_model.pt
│   ├── court_keypoint_detector.pt
│   ├── player_detector.pt
│
├── input_videos/                     # User video input folder
│   ├── video.mp4
│
├── output_videos/                    # Processed results (LFS tracked)
│   ├── result1.avi
│   ├── output_result.avi
│
├── utils/                            # Core utility functions
├── trackers/                         # DeepSORT + Kalman filters
├── configs/                          # Thresholds and config files
├── tools/                            # Helper scripts
│
├── pose_estimator.py                 # Pose estimation module
├── shot_detector.py                  # Shot detection module
├── pass_and_interception_detector.py # Passing/interception logic
├── tactical_view_converter           # Tactical view generation
├── speed_and_distance_calculator     # Speed & distance metrics
├── court_keypoint_detector/          # Court line detection
├── team_assigner/                    # Jersey clustering logic
├── drawers/                          # Drawing overlays
├── ball_aquisition/                  # Ball handling module
│
├── stubs/                            # Pickled intermediate data
│   ├── player_tracks.pkl
│   ├── shot_map_stub.pkl
│   ├── passes_stub.pkl
│   ├── player_speeds_stub.pkl
│
├── training_notebooks/               # Model training experiments
└── requirements.txt                  # Python dependencies
⸻

🛠 Installation

✔ 1. Clone the Repository
git clone git@github.com:adithyakr8055/Basketball_Analysis.git
cd Basketball_Analysis
⚠ Models and large videos are stored with Git LFS — ensure Git LFS is installed.

Install Git LFS
git lfs install
git lfs pull

⸻

✔ 2. Create Virtual Environment (Recommended)

macOS / Linux

⸻

✔ 3. Install Dependencies
pip install -r requirements.txt

⸻

▶️ Running the System

Place your game video inside:
input_videos/
Then run:
python main.py --video input_videos/video.mp4
Outputs will appear in:
output_videos/
And analytics JSON/PKL files inside:
stubs/

🧠 Core Modules Explained

🟦 1. Player Detection
	•	YOLO-based model (models/player_detector.pt)
	•	Performs frame-by-frame detections
	•	DeepSORT assigns consistent tracking IDs

🟡 2. Ball Detection
	•	Small YOLO model (models/ball_detector_model.pt)
	•	Ball centroid estimation
	•	Kalman filter smoothing

📐 3. Court Keypoint Detector
	•	Detects intersections of court lines
	•	Computes homography for converting pixel → real-world court coordinates

🧍‍♂️ 4. Pose Estimator
	•	Landmark detection
	•	Pose classification (jumping, shooting, defending)

🎯 5. Shot Detector
	•	Detects shooting motion
	•	Tracks ball trajectory
	•	Determines shot success/failure

🤝 6. Pass & Interception Detector
	•	Distance-based pass detection
	•	Ball possession timeline
	•	Interception events

🏃‍♂️🌡 7. Speed & Distance
	•	Converts position changes into real-world meters
	•	Computes:
	•	Instant speed
	•	Average speed
	•	Total distance covered

🟥🟦 8. Team Assigner
	•	Extracts jersey colors
	•	Clusters with KMeans
	•	OCR to confirm player number

⸻

📈 Example Analytics Output

✔ Player heatmaps

✔ Shot maps

✔ Pass network graph

✔ Possession stats

✔ Player speeds (km/h)

✔ Player distance covered

✔ Tactical overview (top-down view)

⸻

⚠️ Git LFS Notes

The repository uses Git LFS for:
	•	.pt model files (hundreds of MB)
	•	Output videos .avi
	•	Input videos .mp4

If any LFS file is missing, run:
git lfs pull

⸻

🧪 Training

Training notebooks are in:
training_notebooks/
Includes:
	•	YOLO training
	•	OCR experiments
	•	Pose model training
	•	Court keypoint training

⸻

🛠 Troubleshooting

❗ Models not downloaded
git lfs pull

❗ OpenCV not opening video

Install ffmpeg support:
brew install ffmpeg          # macOS
sudo apt install ffmpeg      # Linux

❗ CUDA/GPU Issues

Use CPU-only version OR install correct CUDA runtime.


⸻

🤝 Contribution Guidelines
	1.	Fork repository
	2.	Create feature branch
	3.	Submit PR with detailed description
	4.	Ensure code is formatted with Black
	5.	Add comments for every module

⸻

📜 License

Choose one:
	•	MIT
	•	Apache 2.0
	•	Proprietary (internal use)

(Tell me your preference and I will customize the license section.)

⸻

🎉 Final Notes

This project aims to deliver real-time basketball match analytics using state-of-the-art CV and deep learning techniques.
The modular design allows easy extension, retraining, and debugging.
Massive files are stored via Git LFS for optimal performance.
