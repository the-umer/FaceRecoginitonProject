from ultralytics import YOLO
import cv2

# Load YOLOv8-face model
model = YOLO("yolov8n-face.pt")  # lightweight & fast

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot access webcam")
    exit()

print("✅ YOLOv8 Face Detection Running... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO model
    results = model(frame, stream=True)

    # Draw bounding boxes
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # Draw box + confidence
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("YOLOv8 Face Detection", frame)

    # Quit on q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
