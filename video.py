#date 20 april 2024
import cv2
import numpy as np
import argparse
import tensorflow as tf

# Load TensorFlow Lite model and labels
MODEL_PATH = "model_unquant.tflite"
LABELS_PATH = "labels.txt"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Load labels
with open(LABELS_PATH, 'r') as f:
    labels = f.read().splitlines()

def detect_objects(frame):
    input_data = cv2.resize(frame, (224, 224))
    input_data = input_data.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)

    input_tensor_index = interpreter.get_input_details()[0]['index']
    output_tensor_index = interpreter.get_output_details()[0]['index']

    interpreter.set_tensor(input_tensor_index, input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_tensor_index)
    class_id = np.argmax(output_data)
    confidence = output_data[0, class_id]

    return labels[class_id], confidence

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        class_name, confidence = detect_objects(frame)
        print("Detected object:", class_name)
        print("Confidence:", confidence)

        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform object detection on a video.")
    parser.add_argument("--video", required=True, help="Path to input video.")
    args = parser.parse_args()
    
    main(args.video)
