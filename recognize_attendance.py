
import cv2
import os
import json
import pandas as pd
from datetime import datetime

# Charger le modèle
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

with open("labels.json", "r", encoding="utf-8") as f:
    students = json.load(f)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Simulation : images présentées pendant la séance
session_images = [
    "dataset/Sara/5.jpg",
    "dataset/Amine/5.jpg",
    "dataset/Adam/5.jpg"
]

# Seuil expérimental LBPH
threshold = 60

# Étudiants reconnus
recognized = set()

for image_path in session_images:

    image = cv2.imread(image_path)

    if image is None:
        print("Image not found:", image_path)
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, 1.1, 5, minSize=(50, 50)
    )

    for x, y, w, h in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        label, distance = recognizer.predict(face)

        if distance < threshold:
            name = students[str(label)]
            recognized.add(name)
            print(f"Recognized: {name} | Distance: {distance:.2f}")
        else:
            print(f"Unknown face | Distance: {distance:.2f}")

# Générer la feuille de présence
now = datetime.now()

attendance = []

for name in students.values():

    status = "Present" if name in recognized else "Absent"

    attendance.append({
        "Name": name,
        "Date": now.strftime("%Y-%m-%d"),
        "Time": now.strftime("%H:%M:%S") if status == "Present" else "",
        "Status": status
    })

df = pd.DataFrame(attendance)

df.to_csv("attendance.csv", index=False)

print("\n===== ATTENDANCE REPORT =====")
print(df)

print("\nAttendance saved successfully!")
