from ultralytics import YOLO
import numpy as np
import imutils
import cv2
import os
from datetime import datetime

from config.parameters import WIDTH
from config.parameters import quit_key, frame_name

# Model
model = YOLO(r"configs\shoplifting_wights.pt")

cap = cv2.VideoCapture(0)

# ✅ Folder penyimpanan video
output_folder = "shoplifting_records"
os.makedirs(output_folder, exist_ok=True)

# ✅ Variabel rekam video
writer = None
recording = False
no_detect_counter = 0
NO_DETECT_LIMIT = 50  # stop rekam setelah 50 frame tidak ada shoplifting (~2 detik)

print("[INFO] Sistem berjalan... tekan Q untuk keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=WIDTH)

    results = model.predict(frame, verbose=False, conf=0.3, iou=0.4)
    boxes = results[0].boxes

    shoplifting_count = 0
    normal_count = 0

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            clas = int(box.cls[0])

            if clas == 1:
                # 🔴 SHOPLIFTING
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"SHOPLIFTING {round(conf*100, 1)}%"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                shoplifting_count += 1

            elif clas == 0:
                # 🟢 NORMAL
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                label = f"NORMAL {round(conf*100, 1)}%"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                normal_count += 1

    # ✅ Logic rekam video
    if shoplifting_count > 0:
        no_detect_counter = 0  # reset counter

        if not recording:
            # Mulai rekam — nama file pakai timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_folder, f"shoplifting_{timestamp}.avi")
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(filename, fourcc, 20,
                                     (frame.shape[1], frame.shape[0]))
            recording = True
            print(f"[REKAM] Mulai merekam: {filename}")

    else:
        if recording:
            no_detect_counter += 1
            if no_detect_counter >= NO_DETECT_LIMIT:
                # Stop rekam
                writer.release()
                writer = None
                recording = False
                no_detect_counter = 0
                print("[REKAM] Selesai merekam, video disimpan!")

    # ✅ Tulis frame ke video jika sedang rekam
    if recording and writer is not None:
        writer.write(frame)
        # Tampilkan indikator rekam (lingkaran merah berkedip)
        cv2.circle(frame, (frame.shape[1] - 20, 20), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (frame.shape[1] - 55, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Summary di pojok kiri atas
    total = shoplifting_count + normal_count
    summary = f"Total: {total} | Shoplifting: {shoplifting_count} | Normal: {normal_count}"
    cv2.rectangle(frame, (0, 0), (500, 35), (0, 0, 0), -1)
    cv2.putText(frame, summary, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow(frame_name, frame)

    if cv2.waitKey(1) & 0xFF == ord(quit_key):
        break

# Pastikan writer ditutup saat keluar
if writer is not None:
    writer.release()

cap.release()
cv2.destroyAllWindows()