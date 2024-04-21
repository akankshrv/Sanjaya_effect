# import cv2
# import os
# import time
# import uuid
# import speech_recognition as sr

# def take_voice_input():
#     recognizer = sr.Recognizer()
#     microphone = sr.Microphone()

#     with microphone as source:
#         print("Listening...")
#         recognizer.adjust_for_ambient_noise(source)
#         audio = recognizer.listen(source)

#     try:
#         text = recognizer.recognize_google(audio)
#         return text
#     except sr.UnknownValueError:
#         print("Sorry, I couldn't understand your voice.")
#     except sr.RequestError:
#         print("Sorry, I'm unable to process your request.")

# def to_ret_label():
#     labels = []
#     while True:
#         user_input = take_voice_input()
#         if user_input:
#             labels.append(user_input)
#         return labels

# def face_reco(labels):
#     IMAGES_PATH = r"collection images"
#     num_of_imgs = 15
#     visit=[0]

#     for label in labels:
#         if visit[label]==0:
#             os.makedirs(os.path.join(IMAGES_PATH, label), exist_ok=True)
#             cap = cv2.VideoCapture(0)
#             print('Collecting images for {}'.format(label))
#             time.sleep(5)
#             visit[label]=1
            
#             for imgnum in range(num_of_imgs):
#                 ret, frame = cap.read()
#                 imgname = os.path.join(IMAGES_PATH, label, '{}.jpg'.format(str(uuid.uuid4())))
#                 cv2.imwrite(imgname, frame)
#                 cv2.imshow('frame', frame)
#                 time.sleep(2)
                
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
                    
#             cap.release()
#             cv2.destroyAllWindows()

# # def face_reco(labels):
# #     IMAGES_PATH = r"collection images"
# #     num_of_imgs = 15
# #     visit = [0] * len(labels)  # Initialize visit list to keep track of visited labels

# #     for label in labels:
# #         label_index = labels.index(label)
# #         if visit[label_index] == 0:
# #             os.makedirs(os.path.join(IMAGES_PATH, label), exist_ok=True)
# #             cap = cv2.VideoCapture(0)
# #             print('Collecting images for {}'.format(label))
# #             time.sleep(5)
# #             visit[label_index] = 1  # Mark label as visited
            
# #             for imgnum in range(num_of_imgs):
# #                 ret, frame = cap.read()
# #                 imgname = os.path.join(IMAGES_PATH, label, '{}.jpg'.format(str(uuid.uuid4())))
# #                 cv2.imwrite(imgname, frame)
# #                 cv2.imshow('frame', frame)
# #                 time.sleep(2)
                
# #                 if cv2.waitKey(1) & 0xFF == ord('q'):
# #                     break
                    
# #             cap.release()
# #             cv2.destroyAllWindows()

import cv2
import os
import time
import uuid
import speech_recognition as sr

def take_voice_input():
    try:
        text = input("Type your input: ")
        return text
    except KeyboardInterrupt:
        print("User interrupted input.")
        return None


def to_ret_label():
    labels = []
    while True:
        user_input = take_voice_input()
        if user_input:
            labels.append(user_input)
        return labels

def face_reco(labels):
    IMAGES_PATH = r"collection_images"  # Adjusted directory name
    num_of_imgs = 15
    visit = {label: 0 for label in labels}  # Initialize visit dictionary for each label

    for label in labels:
        if visit[label] == 0:
            os.makedirs(os.path.join(IMAGES_PATH, label), exist_ok=True)
            cap = cv2.VideoCapture(0)
            print('Collecting images for {}'.format(label))
            time.sleep(5)
            visit[label] = 1
            
            for imgnum in range(num_of_imgs):
                ret, frame = cap.read()
                imgname = os.path.join(IMAGES_PATH, label, '{}.jpg'.format(str(uuid.uuid4())))
                cv2.imwrite(imgname, frame)
                cv2.imshow('frame', frame)
                time.sleep(2)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()

def main():
    labels = to_ret_label()
    face_reco(labels)

if __name__ == "__main__":
    main()
