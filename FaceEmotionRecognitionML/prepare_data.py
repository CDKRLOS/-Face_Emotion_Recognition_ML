import os
import cv2
import numpy as np

from utils import get_face_landmarks

data_dir = './data_train'

output = []
for emotion_indx, emotion in enumerate(sorted(os.listdir(data_dir))):
    emotion_dir = os.path.join(data_dir, emotion)
    for image_name in os.listdir(emotion_dir):
        image_path = os.path.join(emotion_dir, image_name)

        print(f"Processing image: {image_path}")

        if not os.path.isfile(image_path):
            print(f"File does not exist: {image_path}")
            continue

        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to read image: {image_path}")
            continue

        face_landmarks = get_face_landmarks(image)

        if len(face_landmarks) == 1404:
            face_landmarks.append(int(emotion_indx))
            output.append(face_landmarks)

np.savetxt('data.txt', np.asarray(output))
