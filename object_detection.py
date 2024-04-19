import cv2
import numpy as np
import tensorflow as tf

# Load your TensorFlow Lite model and labels
MODEL_PATH = "model_unquant.tflite"
LABELS_PATH =  "labels.txt"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Load labels
with open(LABELS_PATH, 'r') as f:
    labels = f.read().splitlines()

# Initialize video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_data = cv2.resize(frame, (224, 224))
    input_data = input_data.astype(np.float32)  # Cast to FLOAT32
    input_data = input_data / 255.0  # Normalize pixel values to [0, 1]
    input_data = np.expand_dims(input_data, axis=0)

    # Perform inference
    input_tensor_index = interpreter.get_input_details()[0]['index']
    output_tensor_index = interpreter.get_output_details()[0]['index']
    interpreter.set_tensor(input_tensor_index, input_data)
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_tensor_index)

    # Process the output
    class_id = np.argmax(output_data)
    confidence = output_data[0, class_id]

    # Display the result
    if confidence >= 0.65:
        class_name = labels[class_id]
        cv2.putText(frame, f"{class_name}: {confidence:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Cannot detect", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
