import cv2
from simple_facerec import SimpleFacerec
import csv
from datetime import datetime
import time

# Initialize face recognizer
sfr = SimpleFacerec()
sfr.load_encoding_images("archive/Faces")

# Initialize video capture
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error: Could not open video stream from webcam.")
    exit()

# Prepare CSV logging
csv_file = open('face_log.csv', mode='a', newline='')
csv_writer = csv.writer(csv_file)

# Write header only if file is empty
if csv_file.tell() == 0:
    csv_writer.writerow(['Name', 'Date', 'Time'])

print("Camera warming up...")
time.sleep(2)  # Wait 2 seconds before starting detection

detected = False
start_time = time.time()

while True:
    ret, frame = video.read()

    if not ret or frame is None:
        print("Failed to grab frame.")
        break

    face_locations, face_names = sfr.detect_known_faces(frame)

    if face_names and not detected:
        for face_loc, name in zip(face_locations, face_names):
            y1, x1, y2, x2 = face_loc

            # Draw box and label on screen
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Log to CSV
            now = datetime.now()
            csv_writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
            print(f"Logged {name} at {now.strftime('%Y-%m-%d %H:%M:%S')}")

            detected = True  # Prevent further logging
            detection_display_start = time.time()

    if detected:
        # Show frame with detection for 5 seconds then break
        cv2.imshow("Face Detected", frame)
        if time.time() - detection_display_start > 5:
            break
    else:
        cv2.imshow("Scanning for Face...", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
csv_file.close()
video.release()
cv2.destroyAllWindows() 