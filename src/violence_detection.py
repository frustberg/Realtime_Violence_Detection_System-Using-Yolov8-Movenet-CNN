import numpy as np, cv2, collections
PREDICTION_HISTORY = collections.deque(maxlen=15)
POSE_HISTORY = collections.deque(maxlen=10)

def validate_model(model, expected_shape):
    if model is None:
        return False
    try:
        dummy_input = np.zeros(expected_shape)
        model.predict(dummy_input, verbose=0)
        return True
    except Exception as e:
        print(f"Model validation failed: {e}")
        return False

class ModelWrapper:
    def __init__(self, model, sequence_length=20, frame_size=(224,224)):
        self.model = model
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.is_compatible = False
        if model is not None:
            expected_shape = (1, sequence_length, frame_size[1], frame_size[0], 3)
            self.is_compatible = validate_model(model, expected_shape)
            if not self.is_compatible:
                try:
                    input_shape = model.input_shape
                    if input_shape and len(input_shape) == 5:
                        _, seq, height, width, _ = input_shape
                        self.sequence_length = seq
                        self.frame_size = (width, height)
                        expected_shape = (1, self.sequence_length, self.frame_size[1], self.frame_size[0], 3)
                        self.is_compatible = validate_model(model, expected_shape)
                except Exception as e:
                    print(f"Error examining model: {e}")

    def preprocess_frame(self, frame):
        frame = cv2.resize(frame, self.frame_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype('float32') / 255.0
        return frame

    def predict(self, frames_list):
        if not self.is_compatible or self.model is None:
            return 0.0
        if len(frames_list) != self.sequence_length:
            return 0.0
        try:
            sequence = np.array(frames_list)
            sequence = np.expand_dims(sequence, axis=0)
            pred = self.model.predict(sequence, verbose=0)[0][0]
            return float(pred)
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.0

def analyze_frame_motion(current_frame, prev_frame):
    if prev_frame is None or current_frame is None:
        return 0.0
    try:
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
        mean_motion = np.mean(magnitude)
        max_motion = np.max(magnitude)
        motion_score = min(mean_motion / 10.0, 1.0) * 0.5 + min(max_motion / 50.0, 1.0) * 0.5
        return motion_score
    except Exception as e:
        print(f"Error in frame motion analysis: {e}")
        return 0.0

def analyze_motion(current_pose, debug_mode=False):
    if len(POSE_HISTORY) < 2:
        POSE_HISTORY.append(current_pose)
        return 0.0
    prev_pose = POSE_HISTORY[-1]
    velocities = []
    for i in range(len(current_pose)):
        if current_pose[i,2] > 0.2 and prev_pose[i,2] > 0.2:
            dx = current_pose[i,0] - prev_pose[i,0]
            dy = current_pose[i,1] - prev_pose[i,1]
            velocity = np.sqrt(dx**2 + dy**2)
            velocities.append(velocity)
    jerks = []
    if len(POSE_HISTORY) >= 3:
        prev_prev = POSE_HISTORY[-2]
        for i in range(len(current_pose)):
            if (current_pose[i,2] > 0.2 and prev_pose[i,2] > 0.2 and prev_prev[i,2] > 0.2):
                prev_dx = prev_pose[i,0] - prev_prev[i,0]
                prev_dy = prev_pose[i,1] - prev_prev[i,1]
                prev_vel = np.sqrt(prev_dx**2 + prev_dy**2)
                curr_dx = current_pose[i,0] - prev_pose[i,0]
                curr_dy = current_pose[i,1] - prev_pose[i,1]
                curr_vel = np.sqrt(curr_dx**2 + curr_dy**2)
                jerk = abs(curr_vel - prev_vel)
                jerks.append(jerk)
    aggressive_pose_score = 0.0
    if current_pose[5,2] > 0.2 and current_pose[6,2] > 0.2:
        shoulder_y = (current_pose[5,1] + current_pose[6,1]) / 2
        if current_pose[9,2] > 0.2 and current_pose[9,1] < shoulder_y:
            aggressive_pose_score += 0.25
        if current_pose[10,2] > 0.2 and current_pose[10,1] < shoulder_y:
            aggressive_pose_score += 0.25
    if (current_pose[5,2] > 0.2 and current_pose[9,2] > 0.2 and current_pose[6,2] > 0.2 and current_pose[10,2] > 0.2):
        left_arm_ext = np.sqrt((current_pose[9,0]-current_pose[5,0])**2 + (current_pose[9,1]-current_pose[5,1])**2)
        right_arm_ext = np.sqrt((current_pose[10,0]-current_pose[6,0])**2 + (current_pose[10,1]-current_pose[6,1])**2)
        shoulder_width = np.sqrt((current_pose[5,0]-current_pose[6,0])**2 + (current_pose[5,1]-current_pose[6,1])**2)
        if shoulder_width > 0:
            if left_arm_ext / shoulder_width > 1.8:
                aggressive_pose_score += 0.25
            if right_arm_ext / shoulder_width > 1.8:
                aggressive_pose_score += 0.25
    POSE_HISTORY.append(current_pose)
    motion_score = 0.0
    if velocities:
        max_velocity = max(velocities)
        norm_velocity = min(max_velocity / 30, 1.0)
        motion_score += norm_velocity * 0.4
        if max_velocity > 20 and debug_mode:
            print(f"High velocity detected: {max_velocity:.2f}, normalized: {norm_velocity:.2f}")
    if jerks:
        max_jerk = max(jerks)
        norm_jerk = min(max_jerk / 15, 1.0)
        motion_score += norm_jerk * 0.4
        if max_jerk > 10 and debug_mode:
            print(f"High jerk detected: {max_jerk:.2f}, normalized: {norm_jerk:.2f}")
    motion_score += aggressive_pose_score * 0.2
    if motion_score > 0.3 and debug_mode:
        print(f"Motion score: {motion_score:.2f} (velocities: {len(velocities)}, jerks: {len(jerks)}, aggressive pose: {aggressive_pose_score:.2f})")
    return motion_score

def get_smoothed_prediction(prediction):
    PREDICTION_HISTORY.append(prediction)
    weights = np.exp(np.linspace(0,1,len(PREDICTION_HISTORY)))
    weighted_avg = np.average(PREDICTION_HISTORY, weights=weights)
    if len(PREDICTION_HISTORY) > 5:
        recent_avg = np.mean(list(PREDICTION_HISTORY)[-5:])
        prev_avg = np.mean(list(PREDICTION_HISTORY)[:-5])
        if recent_avg > prev_avg:
            weighted_avg = weighted_avg * 1.05
        else:
            weighted_avg = weighted_avg * 0.95
    return min(weighted_avg, 1.0)
