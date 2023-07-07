from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cvlib as cv
import pyttsx3
engine = pyttsx3.init()
import cv2
import numpy as np
import os
import face_recognition
import os

        
def emotion():
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        engine.say(f"Failed to open webcam")
        engine.runAndWait()
        exit()
    status, frame = webcam.read()
    face, confidence = cv.detect_face(frame)
    if not status:
        engine.say(f"Failed to read frame from webcam")
        engine.runAndWait()
        exit()
    webcam.release()
    cv2.destroyAllWindows()
    
    model1=load_model('emotion_detection.model')
    EMOTION_LABELS = ["angry", "fearful", "happy", "sad", "surprised"]
    for idx, f in enumerate(face):      
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
        face_crop = np.copy(frame[startY:endY,startX:endX])
        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        conf = model1.predict(face_crop)[0] 
        idx = np.argmax(conf)
        label = EMOTION_LABELS[idx]
    return label

dataset_path = "collection images"

def encode_faces(dataset_path):
    encoded_faces = []
    names = []

   
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)

        if not os.path.isdir(person_folder):
            continue
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)
            
            if len(face_encoding) > 0:
                encoded_faces.append(face_encoding[0])
                names.append(person_name)

    return encoded_faces, names
    
    
def face_reco():
    encodings, names = encode_faces(dataset_path)
    cap = cv2.VideoCapture(0)
    spoken_names = set()
    emo=emotion()

    while True:
        ret, frame = cap.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            distances = face_recognition.face_distance(encodings, face_encoding)
            matched_index = np.argmin(distances)
            matched_name = names[matched_index]
                
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            if matched_name not in spoken_names:
                if matched_name=="akanksh":
                    engine.say("You are looking great, akanksh")
                    engine.setProperty('rate', 150)  
                    engine.runAndWait()
                    spoken_names.add(matched_name)
                elif matched_name=="nandeesh":
                    engine.say(f"{matched_name} is {emo}")
                    engine.setProperty('rate', 150)  
                    engine.runAndWait()
                    spoken_names.add(matched_name)
                else:
                    engine.say(f"a person is {emo}")
                    engine.setProperty('rate', 150)  
                    engine.runAndWait()
           
        cv2.imshow('output', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
   

