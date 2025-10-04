# Real-Time Violence Detection System

## Overview
The system monitors video streams in real-time to detect violent activity using a combination of computer vision and deep learning techniques. It leverages YOLOv8 for detecting people in the frame, MoveNet for estimating human poses, and an LSTM model to analyze motion patterns over time. When violent behavior is detected, the system can generate alerts and log the video or event details, making it suitable for CCTV surveillance and security monitoring. Alerts are generated when violent behavior is detected, and videos can be logged for future reference. The system also integrates MongoDB to store detection events for future analysis.

## How to Run the Project

### 1. Clone the repository
```
git clone https://github.com/frustberg/Realtime_Violence_Detection_System-Using-Yolov8-Movenet-CNN.git
cd Realtime_Violence_Detection_System-Using-Yolov8-Movenet-CNN
```

### 2. Install dependencies
Create a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```
Install required packages:
```
pip install -r requirements.txt
```

### 3. Configure MongoDB (if logging is enabled)
- Start your MongoDB server.
- Update the MongoDB connection URL in the config file (if any).

### 4. Run the system
- To process live camera feed:
```
python main.py --source 0
```
- To process a video file:
```
python main.py --source path_to_video.mp4
```

### 5. Optional parameters
- `--display` : Set to `True` to show the real-time video with detection overlays.
- `--save_video` : Set to `True` to save the processed video with detection annotations.

