import numpy as np
import os
import torch
from PIL import Image
import mobileclip
import cv2
from ultralytics import YOLO
from mDev import mDEV
import time

frame_index = 0

model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained='./checkpoints/mobileclip_s0.pt')
tokenizer = mobileclip.get_tokenizer('mobileclip_s0')

# text = tokenizer(["elephant", "penguin", "tiger", "pufferfish", "red bottle", "watermelon", "cat", "dog"])
text = tokenizer(["granola box", "floor", "chair", "table", "shoes"])

car = mDEV()
cap = cv2.VideoCapture(0)

car.setServo('3', 95)
# Create folder for saved images
save_path = "saved_images"
os.makedirs(save_path, exist_ok=True)

car.move(225, 225, 120)

# # Initialize camera
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

diff_time = 0

# car.setServo('3', )
# Preload text features
with torch.no_grad(), torch.cuda.amp.autocast():
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    while cap.isOpened():
        # print(car.getSonic())
        start_time = time.time()
        print("start time", start_time)
        ret, frame = cap.read()
        if not ret:
            break
        image_path = f"saved_images/image_0.png"
        print(image_path)
        h, w, _ = frame.shape
        frame_roi = frame
        cv2.imwrite(image_path, frame_roi)
        frame_bgr = cv2.cvtColor(frame_roi, cv2.COLOR_RGB2BGR)
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = preprocess(image).unsqueeze(0)

        # with torch.no_grad(), torch.cuda.amp.autocast():
        cv2.imwrite(image_path, frame_roi)
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        cosine_similarity = (100 * image_features @ text_features.T).softmax(dim=-1)
        print(cosine_similarity.shape)
        print(cosine_similarity)
        if (cosine_similarity[0, 0] > 0.49):
            end_time = time.time()
            diff_time = end_time - start_time
            car.move(0,0, 120)
            time.sleep(1)
            car.move(-350, -350, 120)
            time.sleep(0.85*diff_time)
            car.move(300, 300)
            time.sleep(3)
            car.move(0,0)
            car.setBuzzer(100)
            car.setLed(0,1,0)
            time.sleep(1)
            car.setBuzzer(0)
            car.setLed(0,0,0)
            print("Found")
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()