import cv2
import os

output_dir = 'captured_faces'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)  

def capture_images(person_name, max_images=15):
    person_dir = os.path.join(output_dir, person_name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)

    image_count = len(os.listdir(person_dir))
    images_remaining = max_images - image_count
    
    while images_remaining > 0:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Save the captured face
            face_image = os.path.join(person_dir, f'{person_name}_{image_count}.jpg')
            cv2.imwrite(face_image, gray[y:y+h, x:x+w])

            image_count += 1
            images_remaining -= 1

            # Display the image count
            cv2.putText(frame, f'Images Remaining: {images_remaining}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Face Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or images_remaining <= 0:
            break


person_name = input("Enter the name of the person: ")

capture_images(person_name)

cap.release()
cv2.destroyAllWindows()