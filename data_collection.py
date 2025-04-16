import mediapipe as mp
import numpy as np
import cv2
import os
from datetime import datetime

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Create main directory
os.makedirs("collected_data", exist_ok=True)

# Initialize MediaPipe
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

def collect_data_for_label(name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.npy"
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    X = []
    data_size = 0   
    print(f"\n📸 Starting data collection for '{name}'... Press Esc to stop early or wait for 100 samples.")

    while True:
        lst = []
        ret, frm = cap.read()
        if not ret:
            print("❌ Failed to grab frame from camera.")
            break

        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            # Left hand
            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            # Right hand
            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            X.append(lst)
            data_size += 1

        # Draw landmarks
        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        cv2.putText(frm, f"Samples: {data_size}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Data Collection", frm)

        if cv2.waitKey(1) == 27 or data_size >= 100:  # Esc key or 100 samples
            break

    cap.release()
    cv2.destroyAllWindows()

    X = np.array(X)
    np.save(f"collected_data/{filename}", X)
    print(f"✅ Data saved to collected_data/{filename} | Shape: {X.shape}")

# Main loop
while True:
    label = input("\n🔤 Enter the name of the data (or type 'exit' to quit): ").strip().lower()
    if label == 'exit':
        print("👋 Exiting program.")
        break

    collect_data_for_label(label)

    cont = input("\n🔁 Do you want to collect data for another label? (y/n): ").strip().lower()
    if cont != 'y':
        print("👋 Done collecting all data. Goodbye!")
        break
