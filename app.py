# import sys, cv2, time, numpy as np, csv, os, sqlite3
# from datetime import datetime
# from simple_facerec import SimpleFacerec
# from keras.models import model_from_json
# from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
# from PySide6.QtCore import QTimer, Qt
# from PySide6.QtGui import QImage, QPixmap

# # — 1. Database & CSV Setup with 'punch' column —
# conn = sqlite3.connect('attendance.db')
# cursor = conn.cursor()
# cursor.execute('''
# CREATE TABLE IF NOT EXISTS attendance (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     name TEXT, emotion TEXT, time TEXT, punch TEXT
# )''')
# conn.commit()

# csv_file = 'attendance_log.csv'
# if not os.path.isfile(csv_file):
#     with open(csv_file, 'w', newline='') as f:
#         csv.writer(f).writerow(['name', 'emotion', 'time', 'punch_type'])

# # — 2. Load Face Recognition & Emotion Models —
# sfr = SimpleFacerec()
# sfr.load_encoding_images("archive/Faces")
# with open("emotiondetector.json", "r") as f:
#     emo_model = model_from_json(f.read())
# emo_model.load_weights("emotiondector.h5")
# emo_model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
# labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# def preprocess_face(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     resized = cv2.resize(gray, (48, 48))
#     return resized.reshape(1, 48, 48, 1) / 255.0

# # — 3. Qt GUI: main window with buttons and video —
# class MainApp(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.punch_type = None
#         self.init_ui()
#         self.logged = set()
#         self.stable = {}
#         self.MIN_STABLE = 5
#         self.today = datetime.now().strftime('%Y-%m-%d')
#         self.cap = cv2.VideoCapture(0)
#         self.timer = QTimer()
#         self.timer.timeout.connect(self.update_frame)

#     def init_ui(self):
#         self.setWindowTitle("Punch In/Out")
#         layout = QVBoxLayout()
#         self.label = QLabel("Choose punch type:")
#         layout.addWidget(self.label)

#         btn_in = QPushButton("Punch In")
#         btn_in.clicked.connect(lambda: self.start("IN"))
#         layout.addWidget(btn_in)

#         btn_out = QPushButton("Punch Out")
#         btn_out.clicked.connect(lambda: self.start("OUT"))
#         layout.addWidget(btn_out)

#         self.video_label = QLabel()
#         self.video_label.setAlignment(Qt.AlignCenter)
#         layout.addWidget(self.video_label)

#         btn_exit = QPushButton("Exit")
#         btn_exit.clicked.connect(self.close_app)
#         layout.addWidget(btn_exit)

#         self.setLayout(layout)
#         self.resize(640, 600)

#     def start(self, punch):
#         self.punch_type = punch
#         self.label.setText(f"PUNCH TYPE: {punch}")
#         self.timer.start(100)

#     def update_frame(self):
#         ret, frame = self.cap.read()
#         if not ret:
#             return
#         h, w = frame.shape[:2]
#         faces, names = sfr.detect_known_faces(frame)

#         for (t, r, b, l), name in zip(faces, names):
#             if name == "Unknown":
#                 cv2.rectangle(frame, (l, t), (r, b), (0,0,255), 2)
#                 cv2.putText(frame, "Unknown", (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
#                 continue

#             cursor.execute(
#                 "SELECT 1 FROM attendance WHERE name=? AND punch=? AND DATE(time)=?",
#                 (name, self.punch_type, self.today)
#             )
#             if cursor.fetchone():
#                 continue

#             cx, cy = (l+r)//2, (t+b)//2
#             centered = abs(cx - w//2) < w*0.2 and abs(cy - h//2) < h*0.2
#             self.stable[name] = self.stable.get(name, 0) + (1 if centered else -self.stable.get(name, 0))

#             cv2.rectangle(frame, (l, t), (r, b), (0,255,0), 2)
#             cv2.putText(frame, name, (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

#             if self.stable[name] >= self.MIN_STABLE:
#                 roi = frame[t:b, l:r]
#                 if roi.size == 0:
#                     continue
#                 emo = labels[np.argmax(emo_model.predict(preprocess_face(roi)))]
#                 ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

#                 cursor.execute(
#                     "INSERT INTO attendance (name, emotion, time, punch) VALUES (?, ?, ?, ?)",
#                     (name, emo, ts, self.punch_type)
#                 )
#                 conn.commit()
#                 with open(csv_file, 'a', newline='') as f:
#                     csv.writer(f).writerow([name, emo, ts, self.punch_type])

#                 self.logged.add(name)
#                 cv2.putText(frame, f"{self.punch_type} done: {name}", (30, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
#                 cv2.imshow("Captured", frame)
#                 cv2.waitKey(1000)

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         img = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
#         self.video_label.setPixmap(QPixmap.fromImage(img))

#     def close_app(self):
#         self.timer.stop()
#         self.cap.release()
#         conn.close()
#         self.close()

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     main = MainApp()
#     main.show()
#     sys.exit(app.exec())


# FINAL WORKING CODE 
import sys,cv2 ,numpy as np, csv, os, sqlite3
from datetime import datetime
from threading import Thread
from queue import Queue
from simple_facerec import SimpleFacerec
from keras.models import model_from_json
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap

# — 1. Database & CSV Setup —
conn = sqlite3.connect('attendance.db', timeout=10.0, check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS attendance (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT, emotion TEXT, time TEXT, punch TEXT
)''')
conn.commit()

csv_file = 'attendance_log.csv'
if not os.path.isfile(csv_file):
    with open(csv_file, 'w', newline='') as f:
        csv.writer(f).writerow(['name','emotion','time','punch'])

csv_entries = set()
today_str = datetime.now().strftime('%Y-%m-%d')
with open(csv_file, 'r', newline='') as f:
    for row in csv.reader(f):
        if len(row) == 4 and row[2].startswith(today_str):
            csv_entries.add((row[0], row[3]))

# — 2. Load Face & Emotion Models —
sfr = SimpleFacerec()
sfr.load_encoding_images("archive/Faces")
with open("emotiondetector.json", "r") as f:
    emo_model = model_from_json(f.read())
emo_model.load_weights("emotiondector.h5")
emo_model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
labels = ['angry','disgust','fear','happy','neutral','sad','surprise']

def preprocess_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48,48))
    return resized.reshape(1,48,48,1) / 255.0

# — 3. Async Logger Thread —
log_q = Queue()
def logger_worker():
    while True:
        name, emo, ts, punch = log_q.get()
        if name is None:
            break
        cursor.execute(
            "INSERT INTO attendance(name,emotion,time,punch) VALUES(?,?,?,?)",
            (name, emo, ts, punch)
        )
        conn.commit()
        with open(csv_file, 'a', newline='') as f:
            csv.writer(f).writerow([name, emo, ts, punch])
        log_q.task_done()

Thread(target=logger_worker, daemon=True).start()

# — 4. Main GUI App with Overlay Message —
class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.punch = None
        self.stable = {}
        self.MIN_STABLE = 15
        self.init_ui()
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self, interval=100)
        self.timer.timeout.connect(self.update_frame)

    def init_ui(self):
        self.setWindowTitle("Attendance Punch")
        self.setStyleSheet("""
            QWidget { background-color: #f5f5f5; font-family: Arial; }
            QPushButton { background: #337ab7; color: white; padding: 8px 20px; border: none; border-radius: 4px; font-size: 14px; }
            QPushButton:hover { background: #286090; }
            QPushButton#exitBtn { background: #d9534f; }
            QPushButton#exitBtn:hover { background: #c9302c; }
            QLabel#modeLabel { font-size: 18px; font-weight: bold; }
            QLabel#overlay { color: white; font-size: 24px; background-color: rgba(0,0,0,150); padding: 12px; border-radius: 6px; }
        """)

        main_layout = QVBoxLayout(self)

        # Top-left punch buttons
        top_bar = QHBoxLayout()
        self.in_btn = QPushButton("Punch In")
        self.in_btn.clicked.connect(lambda: self.set_mode("In"))
        top_bar.addWidget(self.in_btn)

        self.out_btn = QPushButton("Punch Out")
        self.out_btn.clicked.connect(lambda: self.set_mode("Out"))
        top_bar.addWidget(self.out_btn)

        top_bar.addStretch()
        main_layout.addLayout(top_bar)

        # Mode label centered
        self.mode_label = QLabel("", objectName="modeLabel", alignment=Qt.AlignCenter)
        main_layout.addWidget(self.mode_label)

        # Enlarged video feed (800×600)
        self.video_label = QLabel(alignment=Qt.AlignCenter)
        self.video_label.setFixedSize(800, 600)
        self.video_label.setStyleSheet("background-color: black;")
        main_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        # Overlay label for temporary messages (parented to video_label)
        self.overlay = QLabel(self.video_label, objectName="overlay")
        self.overlay.setAlignment(Qt.AlignCenter)
        self.overlay.hide()

        # Exit button bottom-right
        self.exit_btn = QPushButton("Exit", objectName="exitBtn")
        self.exit_btn.clicked.connect(self.close_app)
        main_layout.addWidget(self.exit_btn, alignment=Qt.AlignRight)

        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)
        self.resize(850, 800)

    def set_mode(self, mode):
        self.punch = mode
        self.stable.clear()
        self.mode_label.setText(f"Mode → {mode}")
        self.timer.start(0)

    def show_overlay(self, text, duration=1500):
        self.overlay.setText(text)
        self.overlay.adjustSize()
        x = (self.video_label.width() - self.overlay.width()) // 2
        self.overlay.move(x, 30)
        self.overlay.show()
        QTimer.singleShot(duration, self.overlay.hide)

    def update_frame(self):
        if self.punch is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            return

        small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        faces, names = sfr.detect_known_faces(small)
        h, w = frame.shape[:2]

        for (t_,r_,b_,l_), name in zip(faces, names):
            if (name, self.punch) in csv_entries:
                continue
            t, r, b, l = [int(v*2) for v in (t_,r_,b_,l_)]
            color = (0,0,255) if name == "Unknown" else (0,255,0)
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            cv2.putText(frame, name, (l, t-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if name == "Unknown":
                continue

            cx, cy = (l+r)//2, (t+b)//2
            if abs(cx - w//2) < 0.2*w and abs(cy - h//2) < 0.2*h:
                self.stable[name] = self.stable.get(name, 0) + 1
            else:
                self.stable[name] = 0

            if self.stable[name] > self.MIN_STABLE +15:
                secs = (self.stable[name] - self.MIN_STABLE)//5 + 1
                cv2.putText(frame,
                            f"Logging in {3-secs}...",
                            (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,255,0), 2, cv2.LINE_AA)

            if self.stable[name] >= self.MIN_STABLE + 15:
                roi = frame[t:b, l:r]
                if roi.size == 0:
                    continue
                emo = labels[np.argmax(emo_model.predict(preprocess_face(roi)))]
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                log_q.put((name, emo, ts, self.punch))
                csv_entries.add((name, self.punch))

                # Show overlay confirmation
                self.show_overlay(f"{self.punch} DONE: {name}")

                self.stable[name] = -999

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def close_app(self):
        self.timer.stop()
        self.cap.release()
        log_q.put((None, None, None, None))
        conn.close()
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())










