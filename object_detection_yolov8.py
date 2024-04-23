import cv2
import numpy as np
import pyttsx3

net = cv2.dnn.readNetFromDarknet("/home/akanksh/Project1/Vision-Up/yolov3-tiny.cfg", "/home/akanksh/Project1/Vision-Up/yolov3-tiny.weights")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
frames_since_last_detection = {}
previous_detected_objects=[]
output_layers = net.getUnconnectedOutLayersNames()
detected_objects = set()
engine = pyttsx3.init()

def post_process(frame, outputs, conf_threshold, nms_threshold):
    height, width, _ = frame.shape
    boxes = []
    confidences = []
    class_ids = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w // 2
                y = center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    return [boxes[i] for i in indices], [confidences[i] for i in indices], [class_ids[i] for i in indices]

def object_detection(cap):
    while(True):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        ret, frame = cap.read()
        conf_threshold = 0.5  # Confidence threshold for filtering out weak detections
        nms_threshold = 0.4   # Non-maximum suppression threshold for eliminating overlapping boxes

        # Detect objects in the frame
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True)
        net.setInput(blob)
        outputs = net.forward(output_layers)
        boxes, confidences, class_ids = post_process(frame, outputs, conf_threshold, nms_threshold)

        # Update frames_since_last_detection
        for i in range(len(frames_since_last_detection)):
            object_name = list(frames_since_last_detection.keys())[i]
            if object_name in detected_objects:
                frames_since_last_detection[object_name] = 0
            else:
                frames_since_last_detection[object_name] += 1

        # Check for new detections and speak the object name
    # Check for new detections and speak the object name
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        for i in range(len(indices)):
            index = indices[i]
            class_id = class_ids[index]
            object_name = classes[class_id]
            if object_name not in detected_objects:
                detected_objects.add(object_name)
                engine.say(f"{object_name} detected")
                engine.runAndWait()
            frames_since_last_detection[object_name] = 0

        # Draw bounding boxes around the detected objects
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        if len(indices) == 0:
            continue
        for i in indices:
            i = indices[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)


        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

