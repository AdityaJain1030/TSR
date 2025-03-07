from mDev import mDEV
import cv2
import mobileclip
import os
import torch
from PIL import Image
import time

mdev = mDEV()

model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained='./checkpoints/mobileclip_s0.pt')
tokenizer = mobileclip.get_tokenizer('mobileclip_s0')

def doThing(object_to_check):
    car = mDEV()
    cap = cv2.VideoCapture(0)

    save_path = "saved_images"
    os.makedirs(save_path, exist_ok=True)

    text = tokenizer([object_to_check, "floor", "chair", "table", "shoes"])
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    while cap.isOpened():
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

        cv2.imwrite(image_path, frame_roi)
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        cosine_similarity = (100 * image_features @ text_features.T).softmax(dim=-1)
        print(cosine_similarity.shape)
        print(cosine_similarity)
        
        if cosine_similarity[0, 0] > 0.49:
            # car.setBuzzer(100)
            time.sleep(1)
            car.setLed(1, 0, 0)
            print(f"Found {object_to_check}")
        else:
            print(f"Did not find {object_to_check}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

if __name__ == "__main__":
    object_to_check = input("Enter the object you want to check: ")
    doThing(object_to_check)