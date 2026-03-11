from ultralytics import YOLO

# Mulai dari model YOLOv8 kecil yang sudah pretrained
model = YOLO("yolov8n.pt")  # auto download ~6MB

results = model.train(
    data="data.yaml",        # lokasi dataset
    epochs=50,               # jumlah training (lebih banyak = lebih akurat)
    imgsz=640,               # ukuran gambar
    batch=8,                 # sesuaikan dengan RAM laptop
    name="shoplifting_v2",   # nama folder hasil training
    patience=10,             # stop jika tidak ada peningkatan
    device="cpu"             # pakai CPU (ganti "0" jika punya GPU)
)

print("Training selesai!")
print(f"Model tersimpan di: runs/detect/shoplifting_v2/weights/best.pt")