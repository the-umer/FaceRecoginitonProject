import os
import cv2
import torch
import numpy as np
from retinaface import RetinaFace
from facenet_pytorch import InceptionResnetV1

# Initialize device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("Using device:", device)

# Load FaceNet model
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

faces_dir = "faces"
embed_dir = "embeddings"
os.makedirs(embed_dir, exist_ok=True)


# -------------------------------------------------------
# SAFE FACE DETECTION FUNCTION (DOES NOT CRASH)
# -------------------------------------------------------
def detect_and_crop_face(img):
    try:
        detections = RetinaFace.detect_faces(img)

        # If no face detected OR invalid type
        if not isinstance(detections, dict) or len(detections) == 0:
            return None

        # Use first detected face
        key = list(detections.keys())[0]
        x1, y1, x2, y2 = detections[key]["facial_area"]

        # Crop the face
        face = img[y1:y2, x1:x2]
        return face

    except Exception as e:
        print("⚠️ RetinaFace error:", e)
        return None


# -------------------------------------------------------
# SAFE EMBEDDING EXTRACTION FUNCTION (SKIPS BAD IMAGES)
# -------------------------------------------------------
def get_embedding(img_path):
    print("Reading:", img_path)

    img = cv2.imread(img_path)
    if img is None:
        print("❌ Cannot read:", img_path)
        return None

    # Detect face
    face = detect_and_crop_face(img)
    if face is None:
        print("❌ No face detected:", img_path)
        return None

    # Resize and convert to RGB
    try:
        face = cv2.resize(face, (160, 160))
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    except:
        print("❌ Resize/Color error:", img_path)
        return None

    # Convert to tensor
    face_tensor = torch.tensor(face_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
    face_tensor = face_tensor.unsqueeze(0).to(device)

    # Extract embedding
    embedding = resnet(face_tensor).detach().cpu().numpy()[0]
    return embedding


# -------------------------------------------------------
# MAIN SCRIPT
# -------------------------------------------------------
people = os.listdir(faces_dir)
print("People folders found:", people)

for person in people:
    print("\nProcessing person:", person)

    person_folder = os.path.join(faces_dir, person)
    if not os.path.isdir(person_folder):
        print("Skipping (not a folder):", person)
        continue

    embeddings = []

    for img_name in os.listdir(person_folder):
        print("Image file:", img_name)
        img_path = os.path.join(person_folder, img_name)

        emb = get_embedding(img_path)
        if emb is not None:
            embeddings.append(emb)

    embeddings = np.array(embeddings)

    # Save only if we have embeddings
    np.save(os.path.join(embed_dir, f"{person}.npy"), embeddings)
    print(f"✅ Saved embeddings for {person}: {embeddings.shape}")
