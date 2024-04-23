import requests
import cv2
import numpy as np

api_endpoint = "https://api.groq.com/openai/v1/chat/completions"
auth_token = "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"

cap = cv2.VideoCapture(0)  # 0 represents the default webcam, change if needed

if not cap.isOpened():
    print("Error: Unable to access the webcam")
    exit()

ret, frame = cap.read()

if not ret:
    print("Error: Unable to capture frame")
    cap.release()
    exit()

# Release the webcam
cap.release()

# Convert the frame to a binary blob
image_blob = cv2.imencode(".jpg", frame)[1].tobytes()

# Set the question to ask on the image
question = "What is the object in the image?"

# Set the API request headers and body
headers = {"Authorization": f"Bearer {auth_token}", "Content-Type": "application/octet-stream"}
body = {"image": image_blob, "question": question}

# Send the API request
response = requests.post(f"{api_endpoint}/v1/analyze", headers=headers, data=image_blob)

# Check the response status code
if response.status_code == 200:
    print("Image analyzed successfully!")
else:
    print("Error analyzing image:", response.text)

# Print the response JSON
print(response.json())
