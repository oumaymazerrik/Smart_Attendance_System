
import cv2
import os
import json

# Charger le modèle entraîné
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# Charger les noms
with open("labels.json", "r", encoding="utf-8") as f:
    students = json.load(f)

# Détecteur de visages
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

correct = 0
total = 0

for name in sorted(os.listdir("dataset")):

    image_path = os.path.join("dataset", name, "5.jpg")

    image = cv2.imread(image_path)

    if image is None:
        print(f"{name}: image introuvable")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, 1.1, 5, minSize=(50, 50)
    )

    if len(faces) == 0:
        print(f"{name}: aucun visage détecté")
        total += 1
        continue

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))

    label, distance = recognizer.predict(face)

    predicted_name = students[str(label)]

    total += 1

    if predicted_name == name:
        correct += 1
        status = "CORRECT"
    else:
        status = "INCORRECT"

    print(
        f"Real: {name} | "
        f"Predicted: {predicted_name} | "
        f"Distance: {distance:.2f} | "
        f"{status}"
    )

print("\n--------------------------")
print(f"Accuracy: {correct}/{total}")

if total > 0:
    print(f"Accuracy (%): {correct / total * 100:.2f}%")
