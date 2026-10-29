
from deepface import DeepFace
import numpy as np
import matplotlib.pyplot as plt

students = ["Sara", "Yassine", "Amine", "Lina", "Adam"]

distances = np.zeros((5, 5))

for i, student1 in enumerate(students):
    for j, student2 in enumerate(students):

        result = DeepFace.verify(
            img1_path=f"dataset/{student1}/1.jpg",
            img2_path=f"dataset/{student2}/5.jpg",
            model_name="Facenet512",
            detector_backend="opencv"
        )

        distances[i, j] = result["distance"]

        print(
            f"{student1} vs {student2}: "
            f"{result['distance']:.4f}"
        )

plt.figure(figsize=(8, 6))
plt.imshow(distances, cmap="Blues")
plt.colorbar(label="Cosine Distance")

plt.xticks(range(5), students)
plt.yticks(range(5), students)

for i in range(5):
    for j in range(5):
        plt.text(
            j, i,
            f"{distances[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.title("FaceNet512 - Face Similarity Matrix")
plt.tight_layout()
plt.savefig("face_similarity_matrix.png")
plt.show()
