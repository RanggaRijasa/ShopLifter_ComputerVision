# test_detect.py
from ultralytics import YOLO
import cv2
import imutils

model = YOLO(r"configs\shoplifting_wights.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = imutils.resize(frame, width=640)
    
    # Confidence sangat rendah dulu untuk test
    results = model.predict(frame, verbose=True, conf=0.1)
    
    # Print hasil deteksi di terminal
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            print(f"Terdeteksi - Kelas: {int(box.cls[0])} | Conf: {round(float(box.conf[0])*100, 1)}%")
    else:
        print("Tidak ada deteksi...")
    
    cv2.imshow("Test Deteksi", results[0].plot())
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()