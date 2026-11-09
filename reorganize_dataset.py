"""重组数据集：从 fire_yolo_export 扁平结构划分 80/20 train/val 到 dataset_root"""
import os
import shutil
import random

BASE = os.path.dirname(os.path.abspath(__file__))
SRC_IMG = os.path.join(BASE, "fire_yolo_export", "images")
SRC_LBL = os.path.join(BASE, "fire_yolo_export", "labels")

DST_ROOT = os.path.join(BASE, "dataset_root")
DST_IMG_TRAIN = os.path.join(DST_ROOT, "images", "train")
DST_IMG_VAL   = os.path.join(DST_ROOT, "images", "val")
DST_LBL_TRAIN = os.path.join(DST_ROOT, "labels", "train")
DST_LBL_VAL   = os.path.join(DST_ROOT, "labels", "val")

VAL_RATIO = 0.2
SEED = 42

# 1. 收集所有图片文件名
all_images = sorted(f for f in os.listdir(SRC_IMG) if f.lower().endswith((".jpg", ".png", ".jpeg")))
print(f"Total images: {len(all_images)}")

# 2. 随机划分
random.seed(SEED)
random.shuffle(all_images)
val_count = int(len(all_images) * VAL_RATIO)
val_set = set(all_images[:val_count])
train_set = set(all_images[val_count:])
print(f"Train: {len(train_set)}, Val: {len(val_set)}")

# 3. 清空 dataset_root 的 train/val 目录
for d in [DST_IMG_TRAIN, DST_IMG_VAL, DST_LBL_TRAIN, DST_LBL_VAL]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)

# 4. 复制文件
missing_labels = 0
for img in all_images:
    is_val = img in val_set
    dst_img_dir = DST_IMG_VAL if is_val else DST_IMG_TRAIN
    dst_lbl_dir = DST_LBL_VAL if is_val else DST_LBL_TRAIN

    # 复制图片
    shutil.copy2(os.path.join(SRC_IMG, img), os.path.join(dst_img_dir, img))

    # 复制标注（可能不存在 = 负样本）
    stem = os.path.splitext(img)[0]
    lbl_file = stem + ".txt"
    lbl_path = os.path.join(SRC_LBL, lbl_file)
    if os.path.exists(lbl_path):
        shutil.copy2(lbl_path, os.path.join(dst_lbl_dir, lbl_file))
    else:
        missing_labels += 1

print(f"Images without labels (negative samples): {missing_labels}")

# 5. 统计结果
for split, img_dir, lbl_dir in [
    ("train", DST_IMG_TRAIN, DST_LBL_TRAIN),
    ("val",   DST_IMG_VAL,   DST_LBL_VAL),
]:
    imgs = len(os.listdir(img_dir))
    lbls = len(os.listdir(lbl_dir))
    print(f"  {split}: {imgs} images, {lbls} labels, {imgs - lbls} negatives")

print("Done.")
