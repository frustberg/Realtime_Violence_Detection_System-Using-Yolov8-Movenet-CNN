import os, urllib.request, cv2, numpy as np, tensorflow as tf
MOVENET_MODEL_PATH = "movenet_lightning.tflite"
MOVENET_MODEL_URL = "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite"

movenet_available = False
interpreter = None
input_details = None
output_details = None

def download_movenet_model():
    global movenet_available
    try:
        if not os.path.exists(MOVENET_MODEL_PATH):
            headers = {'User-Agent': 'Mozilla/5.0'}
            req = urllib.request.Request(MOVENET_MODEL_URL, headers=headers)
            with urllib.request.urlopen(req) as response, open(MOVENET_MODEL_PATH, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
            movenet_available = True
            return True
        else:
            movenet_available = True
            return True
    except Exception as e:
        print(f"Error downloading MoveNet model: {e}")
        movenet_available = False
        return False

def init_interpreter():
    global interpreter, input_details, output_details, movenet_available
    if not movenet_available:
        return False
    try:
        if os.path.exists(MOVENET_MODEL_PATH) and os.path.getsize(MOVENET_MODEL_PATH) > 0:
            interpreter = tf.lite.Interpreter(model_path=MOVENET_MODEL_PATH)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            movenet_available = True
            print("MoveNet interpreter initialized")
            return True
        else:
            movenet_available = False
            return False
    except Exception as e:
        print(f"Error initializing MoveNet interpreter: {e}")
        movenet_available = False
        return False

def detect_pose(image):
    global interpreter, input_details, output_details, movenet_available
    if interpreter is None or not movenet_available:
        return np.zeros((17,3))
    try:
        input_image = cv2.resize(image, (192,192))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = (input_image * 255).astype(np.uint8)
        interpreter.set_tensor(input_details[0]['index'], input_image)
        interpreter.invoke()
        keypoints = interpreter.get_tensor(output_details[0]['index'])
        keypoints = keypoints.reshape((17,3))
        img_h, img_w = image.shape[0], image.shape[1]
        for i in range(keypoints.shape[0]):
            keypoints[i,1] = keypoints[i,1] * img_w
            keypoints[i,0] = keypoints[i,0] * img_h
        result = np.zeros((17,3))
        result[:,0] = keypoints[:,1]
        result[:,1] = keypoints[:,0]
        result[:,2] = keypoints[:,2]
        return result
    except Exception as e:
        print(f"Error in pose detection: {e}")
        return np.zeros((17,3))
