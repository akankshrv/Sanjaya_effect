import torch
import pyttsx3
engine = pyttsx3.init()
import cv2
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
midas.to(device)
midas.eval()
transforms=torch.hub.load('intel-isl/MiDas', 'transforms')
transform=transforms.small_transform
distance_threshold=1.0

def navi(cap):
    prev_command = None
    engine.say("Navigation Mode activated")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    while True:
        ret, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgbatch = transform(img).to(device)
    
        with torch.no_grad():
            prediction = midas(imgbatch)
            prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1), size=img.shape[:2],mode='bicubic', align_corners=False).squeeze()

            output = prediction.cpu().numpy()

        depth_mask = output <= distance_threshold

        if depth_mask.any():
            left_mask = depth_mask[:, :depth_mask.shape[1] // 2]
            right_mask = depth_mask[:, depth_mask.shape[1] // 2:]
            if not left_mask.any():
                if prev_command != "stop, and Take a left step":
                    engine.say("stop, and Take a left step.")
                    engine.runAndWait()
                    prev_command = "stop, and Take a left step"
                else:
                    pass
            elif not right_mask.any():
                if prev_command != "stop, and Take a right step":
                    engine.say("stop, and Take a right step")
                    engine.runAndWait()
                    prev_command = "stop, and Take a right step"
                else:
                    pass
            else:

                if prev_command != "Stop":
                    engine.say("Stop")
                    engine.runAndWait()
                    prev_command = "Stop"
                else:
                    pass
        else:
            prev_command = None
        cv2.imshow('output', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()

