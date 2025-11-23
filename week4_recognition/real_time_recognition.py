import cv2
import torch
import numpy as np
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import joblib
from attendance_utils import mark_attendance

# -----------------------
# DEVICE SETUP
# -----------------------
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("Using device:", device)

# -----------------------
# LOAD MODELS
# -----------------------

   # FAST face detector
yolo_face = YOLO("week4_recognition/yolov8n-face.pt")


resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
knn = joblib.load("week4_recognition/knn_model.pkl")
    # Classifier

# -----------------------
# EXTRACT FACE EMBEDDING
# -----------------------
def get_embedding(face):
    try:
        face = cv2.resize(face, (160, 160))
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        tensor = torch.tensor(face_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        tensor = tensor.unsqueeze(0).to(device)

        emb = resnet(tensor).detach().cpu().numpy()[0]
        return emb
    except:
        return None

# -----------------------
# REAL-TIME LOOP
# -----------------------
cap = cv2.VideoCapture(0)
print("🎥 YOLO Face Recognition running... Press 'q' to quit.")



while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.resize(frame, (640, 480))


    # 1) YOLO FACE DETECTION (FAST)
    results = yolo_face(frame, verbose=False)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])

            if conf < 0.7:
                continue  # skip low confidence boxes

            face = frame[y1:y2, x1:x2]

            # 2) Embedding
            emb = get_embedding(face)
            if emb is None:
                continue

            # 3) Classification
            identity = knn.predict([emb])[0]
            dist, _ = knn.kneighbors([emb], n_neighbors=1, return_distance=True)
            dist = float(dist[0][0])

            # 4) UNKNOWN CHECK
            if dist > 1.0:
                label = "Unknown"
            else:
                label = f"{identity} ({dist:.2f})"
                mark_attendance(identity)

            # 5) Draw bounding box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

    cv2.imshow("Face Recognition (YOLOv8 Face)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("👋 Exiting...")
