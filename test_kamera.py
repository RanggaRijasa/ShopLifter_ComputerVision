from ultralytics import YOLO
import cv2

# Load model
model = YOLO(r'D:\Shoplifting-Detection-using-yolov8\configs\best.pt')

CLASS_NAMES = ['Normal', 'suspicious-Normal']

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam!")
    exit()

print("Webcam started! Press Q to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    results = model.predict(source=frame, conf=0.25, verbose=False)
    annotated = results[0].plot()

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = CLASS_NAMES[cls_id]
        if label == 'suspicious-Normal':
            cv2.putText(annotated, 'ALERT: SUSPICIOUS BEHAVIOR!',
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3)

    cv2.imshow('Shoplifting Detection', annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Stopped.")