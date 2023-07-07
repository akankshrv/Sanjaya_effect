import pyttsx3
import speech_recognition as sr
from face_reco import face_reco
from navigation import navi
from object_detection import object_detection
def main():
    engine = pyttsx3.init()
    recognizer = sr.Recognizer()
    engine.say("Welcome to, Sanjaya effect.")
    engine.runAndWait()

    while True:
        with sr.Microphone() as source:
            engine.say("Hello")
            engine.runAndWait()
            audio = recognizer.listen(source)
        
        try:
            command = recognizer.recognize_google(audio)
            command=str(command)
            print(command)
            
            if "object detection" in command:
                engine.say("Activating ,Object Detection Mode")
                engine.runAndWait()
                object_detection('yolov8l.onnx')
            elif "navigation" in command:
                engine.say("Activating, Navigation Mode")
                engine.runAndWait()
                navi()
            elif "face read" in command:
                engine.say("Activating, Face Reading Mode")
                engine.runAndWait()
                face_reco()
            elif "exit" in command:
                break
            else:
                engine.say("Can you say that again")
                engine.runAndWait()
                
        
        except sr.UnknownValueError:
            engine.say("Can you say that again")
            engine.runAndWait()
        except sr.RequestError as e:
            engine.say("Could not request results from Speech Recognition service: {0}".format(e))
            engine.runAndWait()

if __name__=="__main__":
    main()