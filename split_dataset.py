"""生成 train.txt / val.txt 文件列表，实现 80/20 train/val 划分（不移动任何文件）"""
import os
import random

BASE = os.path.dirname(os.path.abspath(__file__))
SRC_IMG = os.path.join(BASE, "fire_yolo_export", "images")
EXPORT_DIR = os.path.join(BASE, "fire_yolo_export")

VAL_RATIO = 0.2
SEED = 42

# 1. 收集所有图片
all_images = sorted(f for f in os.listdir(SRC_IMG) if f.lower().endswith((".jpg", ".png", ".jpeg")))
print(f"Total images: {len(all_images)}")

# 2. 随机划分
random.seed(SEED)
random.shuffle(all_images)
val_count = int(len(all_images) * VAL_RATIO)
val_list = all_images[:val_count]
train_list = all_images[val_count:]

# 3. 写入文件列表（路径相对于 fire_yolo_export 目录）
train_txt = os.path.join(EXPORT_DIR, "train.txt")
val_txt = os.path.join(EXPORT_DIR, "val.txt")

with open(train_txt, "w", encoding="utf-8") as f:
    for img in train_list:
        f.write(f"./images/{img}\n")

with open(val_txt, "w", encoding="utf-8") as f:
    for img in val_list:
        f.write(f"./images/{img}\n")

# 4. 统计
src_lbl = os.path.join(EXPORT_DIR, "labels")
train_neg = sum(1 for img in train_list if not os.path.exists(os.path.join(src_lbl, os.path.splitext(img)[0] + ".txt")))
val_neg = sum(1 for img in val_list if not os.path.exists(os.path.join(src_lbl, os.path.splitext(img)[0] + ".txt")))

print(f"Train: {len(train_list)} images ({train_neg} negatives)")
print(f"Val:   {len(val_list)} images ({val_neg} negatives)")
print(f"Written: {train_txt}")
print(f"Written: {val_txt}")
