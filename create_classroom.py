
import cv2
import numpy as np
import os

# Étudiants présents dans la classe fictive
present_students = ["Sara", "Amine", "Adam"]

images = []

for name in present_students:
    path = os.path.join("dataset", name, "5.jpg")

    img = cv2.imread(path)

    if img is None:
        print("Image not found:", path)
        continue

    img = cv2.resize(img, (300, 300))
    images.append(img)

if not images:
    raise ValueError("No student images found")

# Créer une image contenant les étudiants
classroom = np.hstack(images)

cv2.imwrite("classroom.jpg", classroom)

cv2.imshow("Virtual Classroom", classroom)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Classroom image created successfully!")
