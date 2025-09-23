**Violence Detection System**
**Overview**

This project is a Real-Time Violence Detection System that integrates: - YOLOv8 for object detection,
                                                                       - MoveNet for human pose estimation,
                                                                       - A custom LSTM classifier for temporal violence recognition,
                                                                       - MongoDB for logging detection events,
                                                                       - OpenCV for video capture, frame processing, and recording.
The system monitors live or recorded video streams and triggers alerts when violent activity is detected.

**Features**
- Real-time detection from cameras or video files.
- Combination of pose (MoveNet) and object context (YOLOv8) features.
- Custom LSTM model trained on violent vs non-violent sequences (60 frames, 224×224 resolution).
- Logs detection events (camera ID, timestamp, probability, snapshot) to MongoDB.
- Saves flagged clips and can trigger email/SMS/webhook alerts.
