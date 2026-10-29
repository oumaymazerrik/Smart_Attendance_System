

from deepface import DeepFace

# Comparer Sara avec ses propres images
same_person = DeepFace.verify(
    img1_path="dataset/Sara/1.jpg",
    img2_path="dataset/Sara/5.jpg",
    model_name="Facenet512",
    detector_backend="opencv"
)

# Comparer Sara avec Lina
different_person = DeepFace.verify(
    img1_path="dataset/Sara/5.jpg",
    img2_path="dataset/Lina/1.jpg",
    model_name="Facenet512",
    detector_backend="opencv"
)

print("\n===== SAME PERSON =====")
print("Sara vs Sara:", same_person["verified"])
print("Distance:", same_person["distance"])

print("\n===== DIFFERENT PERSON =====")
print("Sara vs Lina:", different_person["verified"])
print("Distance:", different_person["distance"])
