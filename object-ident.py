import cv2
import RPi.GPIO as GPIO
import time

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)
GPIO.setup(27, GPIO.OUT)

left = False
right = False

def togglePin(pin, value):
    val = GPIO.LOW
    if value: val = GPIO.HIGH
    GPIO.output(pin, val)
 
        

# List of objects considered as obstructions
obstruction_objects = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "train",
    "truck",
    "boat",
    "bench",
    "traffic light",
    "fire hydrant",
    "street sign",
    "stop sign",
    "parking meter",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "door"
]

# Load class names
classNames = []
classFile = "/home/anshul/Desktop/Object_Detection_Files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Load model configuration and weights
configPath = "/home/anshul/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/anshul/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    if len(objects) == 0: 
        objects = classNames
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    return img, objectInfo


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)

    while True:
        success, img = cap.read()
        frame_height, frame_width, _ = img.shape

        # Define the central detection area (larger box)
        center_x_min, center_x_max = int(frame_width * 0.3), int(frame_width * 0.7)
        center_y_min, center_y_max = int(frame_height * 0.3), int(frame_height * 0.7)
        cv2.rectangle(img, (center_x_min, center_y_min), (center_x_max, center_y_max), (255, 0, 0), 2)

        result, objectInfo = getObjects(img, 0.45, 0.2, objects=obstruction_objects)

        # Determine the best direction to move
        object_in_center = False
        left_area_clear = True
        right_area_clear = True

        for box, className in objectInfo:
            obj_x_min, obj_y_min, obj_width, obj_height = box
            obj_x_max = obj_x_min + obj_width
            obj_y_max = obj_y_min + obj_height

            # Check for collision with the central detection area
            if (obj_x_min < center_x_max and obj_x_max > center_x_min and
                obj_y_min < center_y_max and obj_y_max > center_y_min):
                object_in_center = True

                # Determine which side the object is closer to
                if (obj_x_max + obj_x_min) / 2 < (center_x_min + center_x_max) / 2:
                    left_area_clear = False
                else:
                    right_area_clear = False

        # Display movement suggestion
        if object_in_center:
            if left_area_clear and not right_area_clear:
                cv2.putText(img, "Move LEFT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                togglePin(27, False)
                togglePin(17, True)
            elif right_area_clear and not left_area_clear:
                cv2.putText(img, "Move RIGHT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                togglePin(17, False)
                togglePin(27, True)
            elif left_area_clear and right_area_clear:
                cv2.putText(img, "Move LEFT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                togglePin(17, True)
                togglePin(27, True)
            else:
                cv2.putText(img, "No Clear Path", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(img, "Move FORWARD", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            togglePin(17, False)
            togglePin(27, False)

        # Display the output
        cv2.imshow("Output", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
