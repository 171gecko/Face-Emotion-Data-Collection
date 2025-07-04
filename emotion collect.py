import cv2
import mediapipe as mp
import csv
import os
from collections import defaultdict

# ===== File & Emotion Setup =====
output_dir = r"C:\Users\overl\Documents\face detection test"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "face_mesh_emotion_dataset.csv")

# Define emotions and assign unique keys
emotion_keys = {
    'a': "angry",
    'h': "happy",
    's': "sad",
    'n': "neutral",
    'u': "surprise",  # 'u' for surprise
    'f': "fear",
    'd': "disgust"
}

# Track sample counts per label
sample_counts = defaultdict(int)

# Default starting label
current_label = "neutral"

# ===== Create CSV if not exists =====
if not os.path.exists(output_file):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["label"]
        for i in range(468):  # 468 face mesh points
            header += [f"x{i}", f"y{i}", f"z{i}"]
        writer.writerow(header)

# ===== MediaPipe Setup =====
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ===== Webcam =====
cap = cv2.VideoCapture(0)

print("[INFO] Press keys to set label and '=' to save sample:")
for key, emotion in emotion_keys.items():
    print(f"  Press '{key.upper()}' for {emotion}")

print("[INFO] Press '=' to save a sample with current label.")
print("[INFO] Press ESC to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    key = cv2.waitKey(1) & 0xFF  # Read key here

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw face mesh
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

            # Display current label and sample count
            cv2.putText(frame, f"Label: {current_label} (Samples: {sample_counts[current_label]})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Save sample
            if key == ord('='):
                landmark_data = [current_label]
                for lm in face_landmarks.landmark:
                    landmark_data.extend([lm.x, lm.y, lm.z])

                with open(output_file, 'a', newline='') as f:
                    csv.writer(f).writerow(landmark_data)

                sample_counts[current_label] += 1
                print(f"[SAVED] {current_label} sample, total: {sample_counts[current_label]}")

    else:
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Handle key for label switching or exit
    if 97 <= key <= 122:  # a-z
        pressed_char = chr(key)
        if pressed_char in emotion_keys:
            current_label = emotion_keys[pressed_char]
            print(f"[INFO] Switched to label: {current_label}")

    elif key == 27:  # ESC
        print("[INFO] Exiting...")
        break

    # Show webcam feed
    cv2.imshow("Face Mesh Emotion Data Collection", frame)

cap.release()
cv2.destroyAllWindows()
print("Label counts:")
print(df.iloc[:, -1].value_counts())
