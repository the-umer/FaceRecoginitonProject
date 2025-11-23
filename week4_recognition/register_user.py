import cv2
import os
import argparse
from retinaface import RetinaFace

FACES_DIR = "../week3_embeddings/faces"
os.makedirs(FACES_DIR, exist_ok=True)

def detect_face(img):
    try:
        detections = RetinaFace.detect_faces(img)
        if not isinstance(detections, dict):
            return None

        key = list(detections.keys())[0]
        x1, y1, x2, y2 = detections[key]['facial_area']

        return img[y1:y2, x1:x2]
    except:
        return None


def capture_images(username):
    user_folder = os.path.join(FACES_DIR, username)
    os.makedirs(user_folder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    print("📸 Webcam started. Look at the camera...")
    print("Capturing 20 images...")

    count = 0
    total_required = 20

    while count < total_required:
        ret, frame = cap.read()
        if not ret:
            continue

        face = detect_face(frame)

        if face is not None:
            face = cv2.resize(face, (160, 160))
            path = os.path.join(user_folder, f"{count}.jpg")
            cv2.imwrite(path, face)
            count += 1
            print(f"✔️ Saved: {path} ({count}/{total_required})")

            cv2.putText(frame, f"Captured: {count}/{total_required}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

        else:
            cv2.putText(frame, "No face detected...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Register User", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("❌ Registration aborted.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if count == total_required:
        print("🎉 Registration complete! 20 images captured.")
    else:
        print(f"⚠️ Only {count} images captured.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="User name")
    args = parser.parse_args()

    username = args.name.lower().replace(" ", "_")

    print(f"👤 Registering: {username}")
    capture_images(username)
