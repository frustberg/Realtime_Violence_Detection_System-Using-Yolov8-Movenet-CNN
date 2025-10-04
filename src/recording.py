import cv2, os, time, uuid
from datetime import datetime

class RecordingManager:
    def __init__(self, output_dir, pre_buffer, post_buffer, fps, frame_size, incidents_collection=None, max_recording_time=60):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.pre_buffer = pre_buffer
        self.post_buffer = post_buffer
        self.fps = fps
        self.frame_size = frame_size
        self.incidents_collection = incidents_collection
        self.max_recording_time = max_recording_time

        self.is_recording = False
        self.video_writer = None
        self.recording_start_time = 0
        self.current_recording_path = None
        self.current_recording_id = None

    def start_recording(self, pre_incident_frames, frame, smoothed_prediction=0.0):
        if self.is_recording:
            return None
        self.current_recording_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"violence_{timestamp}_{self.current_recording_id[:8]}.avi"
        self.current_recording_path = os.path.join(self.output_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(self.current_recording_path, fourcc, self.fps, self.frame_size)
        for f in list(pre_incident_frames):
            try:
                self.video_writer.write(f)
            except Exception:
                pass
        try:
            self.video_writer.write(frame)
        except Exception:
            pass
        self.recording_start_time = time.time()
        self.is_recording = True
        if self.incidents_collection is not None:
            record = {
                "incident_id": self.current_recording_id,
                "filename": filename,
                "filepath": self.current_recording_path,
                "timestamp_start": datetime.now(),
                "detected_score": smoothed_prediction,
                "status": "recording"
            }
            try:
                self.incidents_collection.insert_incident(record)
            except Exception:
                pass
        print(f"Started recording: {self.current_recording_path}")
        return self.current_recording_id

    def stop_recording(self):
        if not self.is_recording:
            return
        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception:
                pass
        duration = time.time() - self.recording_start_time
        print(f"Stopped recording after {duration:.2f} seconds: {self.current_recording_path}")
        if self.incidents_collection is not None and self.current_recording_id is not None:
            try:
                filesize_kb = os.path.getsize(self.current_recording_path) / 1024 if os.path.exists(self.current_recording_path) else 0
                self.incidents_collection.update_incident(self.current_recording_id, {
                    "timestamp_end": datetime.now(),
                    "duration_seconds": duration,
                    "status": "completed",
                    "file_size_kb": filesize_kb
                })
            except Exception:
                pass
        self.is_recording = False
        self.video_writer = None
        self.current_recording_path = None
        self.current_recording_id = None
