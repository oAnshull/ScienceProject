import cv2
import numpy as np
import tensorflow as tf

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="ssd_mobilenet_v2_coco.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load class names from coco.names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width, _ = frame.shape

    # Preprocess frame for MobileNet SSD model
    input_shape = input_details[0]['shape']
    input_tensor = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_tensor = np.expand_dims(input_tensor, axis=0)
    input_tensor = (input_tensor.astype(np.float32) / 127.5) - 1.0

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    # Extract detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    class_ids = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    # Define flags for left, right, and forward area clearance
    left_area_clear = True
    right_area_clear = True
    forward_area_clear = True

    # Define central detection area (adjust as needed)
    center_x_min, center_x_max = int(frame_width * 0.3), int(frame_width * 0.7)
    center_y_min, center_y_max = int(frame_height * 0.1), int(frame_height * 0.9)

    # Check detections
    for i in range(len(scores)):
        if scores[i] > 0.5:  # Confidence threshold
            box = boxes[i]
            class_id = int(class_ids[i])

            # Scale bounding box to frame size
            y_min = int(box[0] * frame_height)
            x_min = int(box[1] * frame_width)
            y_max = int(box[2] * frame_height)
            x_max = int(box[3] * frame_width)
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2

            # Check if object is in the central area
            if center_x_min < center_x < center_x_max and center_y_min < center_y < center_y_max:
                forward_area_clear = False

            # Determine if left or right areas are clear
            if center_x < (center_x_min + center_x_max) // 2:
                left_area_clear = False
            else:
                right_area_clear = False

    # Decide direction based on clear areas
    if forward_area_clear:
        print("Move Forward")
    elif left_area_clear and not right_area_clear:
        print("Move Left")
    elif right_area_clear and not left_area_clear:
        print("Move Right")
    elif left_area_clear and right_area_clear:
        print("Move Left")
    else:
        print("No Clear Path")

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
