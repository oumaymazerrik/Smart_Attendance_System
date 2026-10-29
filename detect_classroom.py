
import cv2
import json

# Charger le modèle
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

with open("labels.json", "r", encoding="utf-8") as f:
    students = json.load(f)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

image = cv2.imread("classroom.jpg")

if image is None:
    raise ValueError("Image classroom.jpg introuvable")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.05,
    minNeighbors=5,
    minSize=(50, 50)
)

# Trier les visages de gauche à droite
faces = sorted(faces, key=lambda face: face[0])

recognized_students = set()
threshold = 90

for x, y, w, h in faces:

    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (200, 200))

    label, distance = recognizer.predict(face)

    if distance < threshold:
        name = students[str(label)]
        recognized_students.add(name)
        color = (0, 255, 0)
    else:
        name = "Unknown"
        color = (0, 0, 255)

    print(f"Prediction: {name} | Distance: {distance:.2f}")

    cv2.rectangle(
        image, (x, y), (x+w, y+h), color, 2
    )

    cv2.putText(
        image,
        f"{name} ({distance:.1f})",
        (x, max(25, y-10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

cv2.imwrite("classroom_result.jpg", image)

cv2.imshow("Smart Attendance System", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nRecognized students:", recognized_students)
