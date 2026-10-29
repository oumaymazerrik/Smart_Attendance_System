
import cv2

# Charger le détecteur de visages
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Activer la webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Erreur : caméra inaccessible")
    exit()

while True:
    ret, frame = camera.read()

    if not ret:
        break

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les visages
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    # Dessiner un rectangle autour des visages
    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # Afficher le nombre de visages détectés
    cv2.putText(
        frame,
        f"Faces detected: {len(faces)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Smart Attendance System", frame)

    # Appuyer sur Q pour quitter
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
