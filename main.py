import cv2
import numpy as np
import threading

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize the camera
cap = cv2.VideoCapture(0)
frame = None
running = True

# Flags to store detection results and a lock for synchronization
left_area_clear = True
right_area_clear = True
forward_area_clear = True
lock = threading.Lock()

# Thread to capture frames
def capture_thread():
    global frame, running
    while running:
        ret, frame = cap.read()
        if not ret:
            running = False

# Thread to process the frame and detect objects
def detection_thread():
    global frame, left_area_clear, right_area_clear, forward_area_clear, running

    while running:
        if frame is None:
            continue
        
        # Lock access to the frame and flags
        with lock:
            frame_height, frame_width, _ = frame.shape

            # Define the central rectangle (adjust as needed)
            center_x_min, center_x_max = int(frame_width * 0.3), int(frame_width * 0.7)
            center_y_min, center_y_max = int(frame_height * 0.1), int(frame_height * 0.9)

            # Detect objects using YOLO
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Reset area flags for each frame
            left_area_clear = True
            right_area_clear = True
            forward_area_clear = True

            # Check for objects in the frame
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    # Filter only high confidence detections
                    if confidence > 0.5:
                        center_x = int(detection[0] * frame_width)
                        center_y = int(detection[1] * frame_height)
                        w = int(detection[2] * frame_width)
                        h = int(detection[3] * frame_height)
                        
                        # Check if the object is within the central detection area
                        if center_x_min < center_x < center_x_max and center_y_min < center_y < center_y_max:
                            forward_area_clear = False  # Obstruction in the forward area

                        # Determine if the left or right areas are clear
                        if center_x < (center_x_min + center_x_max) // 2:
                            left_area_clear = False
                        else:
                            right_area_clear = False

# Thread to display directions based on detection
def display_thread():
    global running
    while running:
        # Lock access to the flags
        with lock:
            if forward_area_clear:
                print("forward")
            elif left_area_clear and not right_area_clear:
                print("left")
            elif right_area_clear and not left_area_clear:
                print("right")
            elif left_area_clear and right_area_clear:
                print("left")
            else:
                print("no path")

# Start threads
capture = threading.Thread(target=capture_thread)
detection = threading.Thread(target=detection_thread)
display = threading.Thread(target=display_thread)

capture.start()
detection.start()
display.start()

# Exit loop on 'q' key press
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

# Wait for threads to complete
capture.join()
detection.join()
display.join()

# Release resources
cap.release()
cv2.destroyAllWindows()
