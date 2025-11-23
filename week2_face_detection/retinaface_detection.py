from retinaface import RetinaFace
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot access webcam")
    exit()

print("✅ RetinaFace Detection Running... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # RetinaFace detects faces in RGB, so convert
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    try:
        faces = RetinaFace.detect_faces(rgb_frame)
    except:
        faces = {}

    if isinstance(faces, dict):
        for key in faces.keys():
            identity = faces[key]

            # Extract bounding box
            x1, y1, x2, y2 = identity["facial_area"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, "Face", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("RetinaFace Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
