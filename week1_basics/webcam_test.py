import cv2

# start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not access webcam.")
    exit()

print("✅ Webcam is running. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Webcam Feed", frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
