import cv2
import numpy as np

# Load MobileNet-SSD model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")

# Initialize class labels for COCO dataset
class_labels = ["background", "aeroplane", "bicycle", "bird", "boat", 
                "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", 
                "dog", "horse", "motorbike", "person", "pottedplant", 
                "sheep", "sofa", "train", "tvmonitor"]

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame_height, frame_width, _ = frame.shape

    # Define the central rectangle (adjust as needed)
    center_x_min, center_x_max = int(frame_width * 0.3), int(frame_width * 0.7)
    center_y_min, center_y_max = int(frame_height * 0.1), int(frame_height * 0.9)
    
    # Draw the central detection area
    cv2.rectangle(frame, (center_x_min, center_y_min), (center_x_max, center_y_max), (255, 0, 0), 2)

    # Detect objects using MobileNet-SSD
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    # Initialize flags for left, right, and forward area clearance
    left_area_clear = True
    right_area_clear = True
    forward_area_clear = True

    # Loop over detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Only consider detections with high confidence
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            label = class_labels[class_id]

            # Get coordinates of the detected object
            box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
            (x, y, w, h) = box.astype("int")
            center_x = int((x + w) / 2)
            center_y = int((y + h) / 2)

            # Draw bounding box around detected object
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {int(confidence * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Check if the object is within the central detection area
            if center_x_min < center_x < center_x_max and center_y_min < center_y < center_y_max:
                forward_area_clear = False  # Obstruction in the forward area

            # Determine if the left or right areas are clear
            if center_x < (center_x_min + center_x_max) // 2:
                left_area_clear = False
            else:
                right_area_clear = False

    # Decide which direction to prompt based on clear areas
    if forward_area_clear:
        print("forward")
        cv2.putText(frame, "Move FORWARD", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif left_area_clear and not right_area_clear:
        print("left")
        cv2.putText(frame, "Move LEFT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif right_area_clear and not left_area_clear:
        print("right")
        cv2.putText(frame, "Move RIGHT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif left_area_clear and right_area_clear:
        print("left")
        cv2.putText(frame, "Move LEFT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        print("no path")
        cv2.putText(frame, "No Clear Path", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Blind Cap Detection", frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
