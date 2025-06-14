import cv2
from keras.models import model_from_json
import numpy as np
import time
import csv

# Load model
with open("emotiondetector.json", "r") as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights("emotiondector.h5")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Face detector
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Labels map
labels = {
    0: 'angry', 1: 'disgust', 2: 'fear',
    3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'
}

def extract_feature(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Start webcam
webcam = cv2.VideoCapture(0)

# Wait 2 seconds before starting detection
print("Warming up camera...")
time.sleep(2)

# Open CSV file
with open("emotion_predictions.csv", mode="a", newline="") as log_file:
    csv_writer = csv.writer(log_file)
    if log_file.tell() == 0:
        csv_writer.writerow(["Timestamp", "Emotion"])

    last_logged_time = 0
    cooldown_seconds = 5
    last_logged_emotion = None

    while True:
        success, frame = webcam.read()
        if not success:
            print("Failed to access camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (48, 48))
            input_feature = extract_feature(face)
            prediction = model.predict(input_feature)
            emotion = labels[prediction.argmax()]

            current_time = time.time()
            if (emotion != last_logged_emotion) or (current_time - last_logged_time > cooldown_seconds):
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"Detected Emotion: {emotion} at {timestamp}")
                csv_writer.writerow([timestamp, emotion])
                last_logged_time = current_time
                last_logged_emotion = emotion

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            break  # Handle only the first detected face for simplicity

        # Display emotion
        if last_logged_emotion:
            cv2.putText(frame, last_logged_emotion, (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Detecting Emotion...", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
webcam.release()
cv2.destroyAllWindows()
