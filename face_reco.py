import time
from keras.preprocessing.image import img_to_array
import tensorflow as tf
import pyttsx3
engine = pyttsx3.init()
import cv2
import numpy as np
import os
import face_recognition
import os

    
dataset_path = "collection images"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
 
    
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import pyttsx3

def emotion(status, frame):
    if not status:
        engine = pyttsx3.init()
        engine.say("Failed to read frame from webcam")
        engine.runAndWait()
        return None
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if face is None:
        return None

    model1 = load_model('./emo_detect.h5')
    EMOTION_LABELS = ["angry", "fearful", "happy", "sad", "surprised"]
    label = None
    for f in face:
        conf = model1.predict(f)[0]
        idx = np.argmax(conf)
        label = EMOTION_LABELS[idx]
    return label


def face_recognition(cap):
    # encodings, names = encode_faces(dataset_path)
    status, frame = cap.read()
    engine.say("Face detection Mode activated")
    spoken_names = set()
    emo = emotion(status,frame)
    engine.say(emo)

    while True:
        _, frame = cap.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            distances = face_recognition.face_distance(encodings, face_encoding)
            matched_index = np.argmin(distances)
            matched_name = names[matched_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            if matched_name not in spoken_names:
                if matched_name == "akanksh":
                    engine.say("You are looking great, akanksh")
                elif matched_name == "nandeesh":
                    engine.say(f"{matched_name} is {emo}")
                else:
                    engine.say(f"a person is {emo}")
                engine.setProperty('rate', 150)
                engine.runAndWait()
                spoken_names.add(matched_name)

        cv2.imshow('output', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

