import cv2
import numpy as np
import argparse
import tensorflow as tf

# Load your TensorFlow Lite model and labels
MODEL_PATH = "model_unquant.tflite"
LABELS_PATH = "labels.txt"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Load labels
with open(LABELS_PATH, 'r') as f:
    labels = f.read().splitlines()

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Object Detection Script")
parser.add_argument("--image", required=True, help="Path to input image")
args = parser.parse_args()

# Read input image
image = cv2.imread(args.image)

# Preprocess the image
input_data = cv2.resize(image, (224, 224))
input_data = input_data.astype(np.float32) / 255.0
input_data = np.expand_dims(input_data, axis=0)

# Perform inference
input_tensor_index = interpreter.get_input_details()[0]['index']
output_tensor_index = interpreter.get_output_details()[0]['index']
interpreter.set_tensor(input_tensor_index, input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_tensor_index)

# Process the output
class_id = np.argmax(output_data)
confidence = output_data[0, class_id]
class_name = labels[class_id]

# Draw bounding box
cv2.putText(image, f"{class_name}: {confidence:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Display the image with detected object
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
