import argparse
import torch
import pyttsx3
import numpy as np
from ultralytics.yolo.utils import ROOT, yaml_load
from ultralytics.yolo.utils.checks import check_yaml
from torchvision import transforms
from PIL import Image
import cv2
import onnx

CLASSES = yaml_load(check_yaml('coco128.yaml'))['names']
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
engine = pyttsx3.init()

def object_detection(onnx_model):
    model = cv2.dnn.readNetFromONNX(onnx_model)
    cap = cv2.VideoCapture(0)
    def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = f'{CLASSES[class_id]} ({confidence:.2f})'
        color = colors[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    detections = {}  # Dictionary to store detected objects
    objs=[]
    engine.say("object detection mode activated..")
    engine.say("I detected many objects .let me say one by one")
    engine.runAndWait()
    
    while True:
        ret, original_image = cap.read()
        [height, width, _] = original_image.shape
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image
        scale = length / 640

        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
        model.setInput(blob)
        outputs = model.forward()

        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.25:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2], outputs[0][i][3]]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

        new_detections = {}  # Temporary dictionary to store new detections
        
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            detection = {
                'class_id': class_ids[index],
                'class_name': CLASSES[class_ids[index]],
                'confidence': scores[index],
                'box': box,
                'scale': scale
            }
            class_name = CLASSES[class_ids[index]]
            if class_name not in detections:
                if class_name not in objs:
                    engine.say(class_name)  # Print the object class name
                    engine.runAndWait()
                    objs.append(class_name)
            new_detections[class_name] = detection
            draw_bounding_box(original_image, class_ids[index], scores[index], round(box[0] * scale),
                              round(box[1] * scale), round((box[0] + box[2]) * scale),
                              round((box[1] + box[3]) * scale))

        # Remove objects that are no longer detected
        for obj in list(detections.keys()):
            if obj not in new_detections:
                del detections[obj]

        detections = new_detections

        cv2.imshow('image', original_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return
