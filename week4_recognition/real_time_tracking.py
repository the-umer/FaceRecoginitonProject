import cv2
import torch
import numpy as np
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import joblib
from boxmot import ByteTrack
from attendance_utils import mark_attendance

# ----------------------------------------
# DEVICE SETUP
# ----------------------------------------
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("Using device:", device)

# ----------------------------------------
# LOAD MODELS
# ----------------------------------------
yolo_face = YOLO("yolov8n-face.pt")
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
knn = joblib.load("knn_model.pkl")

# ----------------------------------------
# INITIALIZE TRACKER
# ----------------------------------------
tracker = ByteTrack()

# ----------------------------------------
# GET FACE EMBEDDING
# ----------------------------------------
def get_embedding(face):
    try:
        face = cv2.resize(face, (160, 160))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(face, dtype=torch.float32).permute(2, 0, 1) / 255.0
        tensor = tensor.unsqueeze(0).to(device)
        emb = resnet(tensor).detach().cpu().numpy()[0]
        return emb
    except:
        return None

# ----------------------------------------
# REAL-TIME RECOGNITION + TRACKING LOOP
# ----------------------------------------
cap = cv2.VideoCapture(0)
print("🎥 YOLO + ByteTrack Face Recognition Running... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (640, 480))

    # YOLO detect
    results = yolo_face(frame, verbose=False)[0]

    # Extract detections
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cls = 0  # single class for faces

        if conf > 0.6:
            detections.append([x1, y1, x2 - x1, y2 - y1, conf, cls])

    detections = np.array(detections)

    # Track faces
    tracks = tracker.update(detections, frame)

    # For each tracked object
    for track in tracks:
        track_id = int(track[4])  # Unique ID
        x, y, w, h = track[:4].astype(int)

        face = frame[y:y+h, x:x+w]

        emb = get_embedding(face)
        if emb is None:
            continue

        # Predict with KNN
        pred = knn.predict([emb])[0]
        dist, _ = knn.kneighbors([emb], n_neighbors=1, return_distance=True)
        dist = float(dist[0][0])

        if dist > 1.1:
            label = f"Unknown (ID {track_id})"
        else:
            label = f"{pred} (ID {track_id})"
            mark_attendance(pred)

        # Draw bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("YOLO + ByteTrack Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("👋 Exiting...")
