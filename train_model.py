
import cv2
import numpy as np
import os
import json

dataset_path = "dataset"

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
students = {}

for student_id, name in enumerate(sorted(os.listdir(dataset_path))):

    folder = os.path.join(dataset_path, name)

    if not os.path.isdir(folder):
        continue

    students[student_id] = name
    count = 0

    for image_name in sorted(os.listdir(folder)):

        if image_name == "5.jpg":
            continue

        image_path = os.path.join(folder, image_name)
        image = cv2.imread(image_path)

        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detected = face_cascade.detectMultiScale(
            gray, 1.1, 5, minSize=(50, 50)
        )

        if len(detected) == 0:
            print("No face:", image_path)
            continue

        x, y, w, h = max(detected, key=lambda f: f[2] * f[3])

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        faces.append(face)
        labels.append(student_id)
        count += 1

    print(f"{name}: {count} training images")

if not faces:
    raise ValueError("No faces found in dataset")

recognizer.train(faces, np.array(labels))
recognizer.save("trainer.yml")

with open("labels.json", "w", encoding="utf-8") as file:
    json.dump(students, file, indent=4)

print("\nModel trained successfully!")
print("Total training images:", len(faces))
