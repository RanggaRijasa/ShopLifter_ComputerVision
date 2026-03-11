"""
Script untuk mengecek isi dataset gabungan YOLOv8.
Menampilkan jumlah gambar dan distribusi label per split.
"""

from pathlib import Path
from collections import Counter

# ============================================================
# SESUAIKAN PATH DAN NAMA KELAS
# ============================================================

DATASET_DIR = r'C:\Users\HYPE R Series\Downloads\dataset_gabungan raw'
CLASS_NAMES   = ['Normal', 'shoplifting']

# ============================================================

def cek_split(dataset_dir, split, class_names):
    img_dir = Path(dataset_dir) / split / 'images'
    lbl_dir = Path(dataset_dir) / split / 'labels'

    if not img_dir.exists():
        print(f"  [{split}] folder tidak ditemukan, dilewati.")
        return

    # Hitung gambar
    images = list(img_dir.glob('*'))
    images = [f for f in images if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]

    # Hitung label per kelas
    label_count = Counter()
    total_objects = 0
    for lbl_file in lbl_dir.glob('*.txt'):
        for line in lbl_file.read_text().strip().splitlines():
            if line.strip():
                class_id = int(line.split()[0])
                label_count[class_id] += 1
                total_objects += 1

    print(f"\n  📂 {split.upper()}")
    print(f"     Jumlah gambar  : {len(images)}")
    print(f"     Total objek    : {total_objects}")
    print(f"     Per kelas      :")
    for idx, name in enumerate(class_names):
        count = label_count.get(idx, 0)
        bar = '█' * (count // max(1, total_objects // 20))
        print(f"       [{idx}] {name:20s} : {count:5d}  {bar}")


def main():
    print("=" * 55)
    print("CEK ISI DATASET GABUNGAN YOLOV8")
    print("=" * 55)
    print(f"Folder : {DATASET_DIR}")
    print(f"Kelas  : {CLASS_NAMES}")

    for split in ['train', 'valid', 'test']:
        cek_split(DATASET_DIR, split, CLASS_NAMES)

    print("\n" + "=" * 55)


if __name__ == '__main__':
    main()
