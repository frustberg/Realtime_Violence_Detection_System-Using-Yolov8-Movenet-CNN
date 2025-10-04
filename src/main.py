import cv2, numpy as np, os, time, sys
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from collections import deque
import pygame  

from pose_estimation import download_movenet_model, init_interpreter, detect_pose
from violence_detection import ModelWrapper, analyze_frame_motion, get_smoothed_prediction, analyze_motion
from mongodb_handler import MongoHandler
from recording import RecordingManager
import utils

print("Starting modular violence detection...")

MONGODB_CONNECTION_STRING = "mongodb+srv://frustberg:Abhishek123@cluster1.oqmff.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1"
DB_NAME = "violence_detection"
COLLECTION_NAME = "recorded_incidents"

RECORDING_CONFIG = {
    "output_directory": "recorded_incidents",
    "pre_incident_buffer": 5,
    "post_incident_buffer": 5,
    "video_codec": "XVID",
    "video_fps": 20,
    "max_recording_time": 60,
}

CONFIG = {
    "violence_threshold": 0.6,
    "yolo_confidence": 0.5,
    "sequence_length": 20,
    "frame_size": (224, 224),
    "motion_threshold": 0.4,
    "jerk_threshold": 0.4,
    "alert_cooldown": 3,
    "debug_mode": True
}

os.makedirs(RECORDING_CONFIG["output_directory"], exist_ok=True)

mongo = MongoHandler(MONGODB_CONNECTION_STRING, DB_NAME, COLLECTION_NAME)

violence_model = None
if os.path.exists("violence_detection_model.h5"):
    try:
        violence_model = load_model("violence_detection_model.h5", compile=False)
    except Exception as e:
        try:
            violence_model = load_model("violence_detection_model.h5", compile=False, custom_objects={'batch_shape': None})
        except Exception:
            try:
                import tensorflow as tf
                violence_model = tf.keras.models.load_model("violence_detection_model.h5", compile=False)
            except Exception:
                violence_model = None
else:
    print("WARNING: violence_detection_model.h5 not found. Using motion-only detection.")

model_wrapper = ModelWrapper(violence_model, sequence_length=CONFIG["sequence_length"], frame_size=CONFIG["frame_size"])
if model_wrapper.is_compatible:
    CONFIG["sequence_length"] = model_wrapper.sequence_length
    CONFIG["frame_size"] = model_wrapper.frame_size

try:
    yolo_model = YOLO("yolov8n.pt")
    print("YOLO loaded")
except Exception as e:
    print(f"YOLO error: {e}")
    yolo_model = None

download_movenet_model()
init_interpreter()
utils.init_audio()

pygame.mixer.init()  # 🔔 Initialize audio system
alarm_playing = False  # 🔔 Track alarm state

try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 20)
except Exception as e:
    print(f"Camera error: {e}")
    sys.exit(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps <= 0:
    fps = RECORDING_CONFIG["video_fps"]

frames = deque(maxlen=CONFIG["sequence_length"])
pre_incident_frames = deque(maxlen=int(RECORDING_CONFIG["pre_incident_buffer"] * fps))

record_manager = RecordingManager(
    RECORDING_CONFIG["output_directory"],
    RECORDING_CONFIG["pre_incident_buffer"],
    RECORDING_CONFIG["post_incident_buffer"],
    RECORDING_CONFIG["video_fps"],
    (frame_width, frame_height),
    incidents_collection=mongo,
    max_recording_time=RECORDING_CONFIG["max_recording_time"]
)

is_recording = False
smoothed_prediction = 0.0
motion_score = 0.0
frame_count = 0
alert_triggered = False
last_alert_time = 0
violence_detected = False
violence_stopped_time = 0
prev_frame = None

print("Press 'q' to quit, 't' to test, 'r' to toggle recording")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        frame_count += 1
        current_time = time.time()
        pre_incident_frames.append(frame.copy())

        if frame_count % 2 != 0:
            if record_manager.is_recording and record_manager.video_writer is not None:
                record_manager.video_writer.write(frame)
            continue

        display_frame = frame.copy()
        detected_humans = []
        human_boxes = []

        if yolo_model is not None:
            try:
                results = yolo_model(frame)
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf[0].item()
                        cls = int(box.cls[0])
                        if cls == 0 and conf > CONFIG["yolo_confidence"]:
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                            if x2 > x1 and y2 > y1:
                                human_frame = frame[y1:y2, x1:x2]
                                detected_humans.append(human_frame)
                                human_boxes.append((x1, y1, x2, y2))
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0,255,0), 2)
                                cv2.putText(display_frame, f"Person: {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            except Exception:
                pass

        this_frame_prediction = 0.0
        if detected_humans:
            largest_idx = np.argmax([h.shape[0]*h.shape[1] for h in detected_humans])
            human_frame = detected_humans[largest_idx]
            human_box = human_boxes[largest_idx]

            pose_keypoints = detect_pose(human_frame)
            if pose_keypoints is not None and pose_keypoints.shape[0] == 17 and np.any(pose_keypoints):
                motion_score = analyze_motion(pose_keypoints, debug_mode=CONFIG["debug_mode"])
                for i in range(pose_keypoints.shape[0]):
                    if pose_keypoints[i,2] > 0.2:
                        x = int(pose_keypoints[i,0] + human_box[0])
                        y = int(pose_keypoints[i,1] + human_box[1])
                        cv2.circle(display_frame, (x,y), 5, (0,0,255), -1)
            else:
                motion_score = analyze_frame_motion(human_frame, prev_frame)
                prev_frame = human_frame.copy() if human_frame is not None else prev_frame

            cv2.putText(display_frame, f"Motion: {motion_score:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            if model_wrapper.is_compatible:
                processed = model_wrapper.preprocess_frame(human_frame)
                frames.append(processed)
                if len(frames) == CONFIG["sequence_length"]:
                    try:
                        this_frame_prediction = model_wrapper.predict(list(frames))
                    except Exception:
                        this_frame_prediction = 0.0
            else:
                this_frame_prediction = motion_score

        smoothed_prediction = get_smoothed_prediction(this_frame_prediction)
        violence_detected = smoothed_prediction > CONFIG["violence_threshold"]

        if violence_detected:
            violence_stopped_time = 0
            if not record_manager.is_recording:
                record_manager.start_recording(pre_incident_frames, frame, smoothed_prediction)

        elif record_manager.is_recording:
            if violence_stopped_time == 0:
                violence_stopped_time = current_time
            if current_time - violence_stopped_time > RECORDING_CONFIG["post_incident_buffer"]:
                record_manager.stop_recording()
            if current_time - record_manager.recording_start_time > RECORDING_CONFIG["max_recording_time"]:
                record_manager.stop_recording()

        if violence_detected:
            if not alert_triggered or (current_time - last_alert_time > CONFIG["alert_cooldown"]):
                utils.update_alert_status(True)
                alert_triggered = True
                last_alert_time = current_time
                print(f"⚠️ VIOLENCE DETECTED! Score: {smoothed_prediction:.2f}")

                # 🔔 Play alarm sound
                if not alarm_playing:
                    try:
                        pygame.mixer.music.load("assets/alarm_sound.mp3")
                        pygame.mixer.music.play(-1)
                        alarm_playing = True
                    except Exception as e:
                        print(f"Error playing alarm: {e}")

        else:
            if alert_triggered:
                utils.update_alert_status(False)
                alert_triggered = False
            # 🔕 Stop alarm if no violence
            if alarm_playing:
                pygame.mixer.music.stop()
                alarm_playing = False

        status_color = (0,0,255) if violence_detected else (0,255,0)
        status_text = "VIOLENCE DETECTED" if violence_detected else "Monitoring"
        cv2.putText(display_frame, status_text, (10, frame_height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        score_text = f"Score: {smoothed_prediction:.2f}"
        cv2.putText(display_frame, score_text, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        if record_manager.is_recording:
            rec_time = time.time() - record_manager.recording_start_time
            cv2.putText(display_frame, f"REC {rec_time:.1f}s", (frame_width-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.circle(display_frame, (frame_width-170, 25), 10, (0,0,255), -1)

        cv2.imshow("Violence Detection System", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            break
        elif key == ord('t'):
            print("Test: Simulating violence detection")
            this_frame_prediction = 0.9
        elif key == ord('r'):
            if not record_manager.is_recording:
                print("Manual recording started")
                record_manager.start_recording(pre_incident_frames, frame, smoothed_prediction)
            else:
                print("Manual recording stopped")
                record_manager.stop_recording()

except KeyboardInterrupt:
    print("Keyboard interrupt. Exiting...")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    if record_manager.is_recording:
        record_manager.stop_recording()
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()  # 🔕 Clean shutdown for pygame
    if mongo is not None and hasattr(mongo, 'close'):
        mongo.close()
    print("System shutdown complete")
