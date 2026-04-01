<p align="center">
  <h1 align="center">🚦 Traffic Analyser</h1>
  <p align="center"><strong>Real-time vehicle detection, tracking & intelligent crowd-level alerts powered by YOLO26</strong></p>
  <p align="center"><em>Drop in any road/drone footage — Traffic Analyser counts vehicles, measures flow per minute, and instantly tells you if the road is crowded.</em></p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/YOLO-v26-0099FF?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.4+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/Ultralytics-00B4D8?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge"/>
</p>

---

## 🎯 What is Traffic Analyser?

**Traffic Analyser** is an AI-powered traffic monitoring system that uses the state-of-the-art **YOLO26** object detection model to:

- **Detect & track** every vehicle in a video stream with unique IDs
- **Count crossings** at a virtual line drawn across the road
- **Measure real-time flow** — vehicles crossing per minute (rolling window)
- **Compare against your road's capacity** and raise a crowd alert instantly

You simply specify how many vehicles your road can handle per minute. Traffic Analyser watches the feed and does the rest.

---

## ⚡ Key Features

| Feature | Description |
|---|---|
| **YOLO26 Detection** | Latest YOLO generation — fast, accurate, NMS-free end-to-end inference |
| **Multi-object Tracking** | Unique ID assigned to every vehicle; persists across frames |
| **Virtual Counting Line** | Animated line drawn at frame midpoint; counts each vehicle that crosses |
| **Rolling 60-second Window** | Traffic rate computed over the *last 60 seconds*, not total video time |
| **3-tier Crowd Alert** | Instant visual status: No Crowd / Slightly Crowded / More Crowded |
| **Rich HUD Overlay** | Glass-morphism dark panel with live stats & FPS |
| **Auto-save Output** | Processed video written to `assets/output.mp4` |
| **CPU Friendly** | Runs without a GPU |

---

## 🧠 How Does Traffic Analyser Decide if a Road is Crowded?

This is the core intelligence of the system. Here is the full pipeline:

### Step 1 — You Set the Road Limit
When you launch the app, it asks:
```
Enter road capacity (max vehicles per minute): 30
```
This is **your road's safe throughput** — how many vehicles it can handle per minute without congestion.

### Step 2 — YOLO26 Detects & Tracks Vehicles
Every frame is passed through YOLO26 with `persist=True` tracking. Each detected vehicle gets a **stable unique ID** that is kept consistent across frames using ByteTrack.

### Step 3 — Virtual Counting Line
A horizontal line is drawn at **50% of the frame height**. The system monitors the Y-coordinate of each vehicle across consecutive frames:

```
Previous frame Y < line_Y  AND  Current frame Y >= line_Y  →  Vehicle crossed (top→bottom)
Previous frame Y > line_Y  AND  Current frame Y <= line_Y  →  Vehicle crossed (bottom→top)
```
Each vehicle ID is counted **only once**, no matter how many frames it stays near the line.

### Step 4 — Rolling 60-Second Rate
Every crossing is timestamped. Traffic Analyser maintains a **sliding window** of the last 60 seconds:

```python
# When the window slides, old crossings are dropped:
while crossing_times and now - crossing_times[0] > 60:
    crossing_times.popleft()

per_minute = len(crossing_times)   # live vehicles/min
```

This means the rate is **always live** — if traffic clears up, the alert resets automatically.

### Step 5 — Crowd Level Comparison

```
per_minute  vs  road_limit
─────────────────────────────────────────────────────────
per_minute  <=  road_limit            →  ✅  NO CROWD
road_limit  <  per_minute  <=  limit+10  →  ⚠   SLIGHTLY CROWDED
per_minute  >  road_limit + 10        →  🚨  MORE CROWDED
```

The threshold of **+10** acts as a buffer — small fluctuations won't trigger a false alarm.  
The status banner at the bottom of the video updates every frame in real time.

---

## 📁 Project Structure

```
Traffic-Analyser/
│
├── main.py                       # General inference entry point (CLI)
├── requirements.txt              # Python dependencies
├── README.md
├── .gitignore
│
├── ui/
│   └── traffic_counter.py        # 🚦 Main app — HUD + counting + crowd alerts
│
├── scripts/
│   └── auto_start.py             # Quick training launcher (VisDrone dataset)
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # All hyperparameters & path config
│   ├── trainer.py                # Model fine-tuning pipeline
│   ├── inference.py              # Live inference / detection engine
│   └── utils.py                  # Shared utilities (logging, I/O, draw)
│
├── models/
│   ├── yolo26n.pt                # Base YOLO26 nano weights
│   └── best.pt                   # Fine-tuned weights (generated after training)
│
├── assets/
│   ├── video.mp4                 # Input video for the traffic counter
│   └── output.mp4                # Processed output (auto-generated)
│
└── data/
    └── VisDrone.yaml             # VisDrone dataset config (for training)
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Yuvraj-025/traffic-analyser.git
cd traffic-analyser

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your input video
# Place your road/drone footage at:  assets/video.mp4

# 4. Launch Traffic Analyser
python ui/traffic_counter.py
# → You'll be prompted:  "Enter road capacity (max vehicles per minute):"
# → Type a number, e.g.  30

# 5. Press Q to quit at any time. Output is saved to assets/output.mp4
```

### Other Commands

```bash
# Run general inference / live detection (webcam or video)
python main.py --mode predict --source assets/video.mp4
python main.py --mode predict --source 0           # 0 = webcam

# Fine-tune the model on VisDrone dataset
python scripts/auto_start.py

# Train with full config
python main.py --mode train
```

---

## 🏋️ Model & Training

| Parameter     | Value                         |
|---------------|-------------------------------|
| Model         | YOLO26n (Nano)                |
| Dataset       | VisDrone2019 (~10k images, 6.5 GB) |
| Epochs        | 20 (with early stopping)      |
| Batch Size    | 16                            |
| Image Size    | 640 × 640                     |
| Optimizer     | MuSGD (YOLO26 built-in)       |
| Precision     | Mixed FP16                    |
| Hardware      | NVIDIA 4060                   |

> **Why VisDrone?** It contains 10,000+ drone-captured images of dense urban traffic, making it the ideal dataset for aerial/road vehicle detection.

---

## 🛠️ Tech Stack

| Layer         | Technology                    |
|---------------|-------------------------------|
| Detection     | YOLO26n — Ultralytics         |
| Tracking      | ByteTrack (built into YOLO26) |
| Deep Learning | PyTorch 2.4+                  |
| Vision / HUD  | OpenCV 4+                     |
| Language      | Python 3.12                   |

---

## 🔮 Roadmap

- [ ] Multi-lane counting with per-lane crowd status
- [ ] Speed estimation via perspective transform
- [ ] Web dashboard (Flask/WebSocket) for live monitoring
- [ ] SQLite logging of traffic events
- [ ] Edge deployment on Raspberry Pi 5 + Hailo-8
- [ ] Alert system (email / SMS) when crowd threshold is breached

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
