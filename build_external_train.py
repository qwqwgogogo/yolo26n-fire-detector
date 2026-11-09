# -*- coding: utf-8 -*-
"""
按用户硬约束重建训练集：
  ❌ 官方 1100 张(dateset_root/images/train) 一张都不进训练集
  ✅ 训练集 = 本地已下载的 D-Fire 公开集(全部, 含火焰正样本+阴性)
  ✅ 官方 1100 永远只当测试集(dataset_root/images/train + labels/train)

D-Fire 类别 = {0:fire,1:smoke}，与竞赛一致，无需重映射，label 直接复制。
图像硬链接(同盘 E: 省空间)，label 复制。
文件名兜底：若 D-Fire 图名与官方图名相同则跳过(防泄漏，虽然不同源基本不会重名)。

为控制训练时长与过拟合，阴性(无烟无火背景)抽样上限 NEG_CAP；火焰正样本全保留以保火识别能力。
"""
import os, glob, random, shutil

random.seed(42)
ROOT = "E:/火情监测/火情监测"
SRC = os.path.join(ROOT, "external_datasets", "academic_D-Fire")
OFF_IMG = os.path.join(ROOT, "dataset_root", "images", "train")
OUT = os.path.join(ROOT, "distill_pipeline", "external_train_dataset")
IMG = os.path.join(OUT, "images", "train")
LBL = os.path.join(OUT, "labels", "train")
NEG_CAP = 7000  # 阴性抽样上限

os.makedirs(IMG, exist_ok=True)
os.makedirs(LBL, exist_ok=True)

# 官方图名集合（兜底排除）
off_names = set()
for b in os.listdir(OFF_IMG):
    off_names.add(os.path.splitext(b)[0])

def link(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy(src, dst)

def find_img(label_path):
    d = os.path.dirname(os.path.dirname(label_path))  # .../labels -> parent
    imgdir = os.path.join(d, "images")
    base = os.path.splitext(os.path.basename(label_path))[0]
    for ext in (".jpg", ".png", ".jpeg", ".JPG"):
        cand = os.path.join(imgdir, base + ext)
        if os.path.exists(cand):
            return cand
    return None

items = []  # (img, lbl, has_fire)
for lp in glob.glob(os.path.join(SRC, "**", "labels", "*.txt"), recursive=True):
    base = os.path.splitext(os.path.basename(lp))[0]
    if base in off_names:
        continue  # 兜底排除官方图
    img = find_img(lp)
    if img is None:
        continue
    has_fire = False
    try:
        with open(lp, encoding="utf-8", errors="ignore") as f:
            for line in f:
                p = line.split()
                if len(p) >= 5 and p[0] == "0":
                    has_fire = True
                    break
    except Exception:
        pass
    items.append((img, lp, has_fire))

fire = [x for x in items if x[2]]
nonfire = [x for x in items if not x[2]]
random.shuffle(nonfire)
nonfire = nonfire[:NEG_CAP]
sel = fire + nonfire

for img, lp, _ in sel:
    base = os.path.basename(img)
    link(img, os.path.join(IMG, base))
    shutil.copy(lp, os.path.join(LBL, os.path.splitext(base)[0] + ".txt"))

# yaml
with open(os.path.join(OUT, "dataset.yaml"), "w", encoding="utf-8") as f:
    f.write("path: %s\n" % OUT)
    f.write("train: images/train\n")
    f.write("val: images/train\n")
    f.write("names: {0: fire, 1: smoke}\n")

print("D-Fire 火焰正样本图: %d" % len(fire))
print("D-Fire 阴性图(抽样%d, 总可用%d): %d" % (NEG_CAP, len([x for x in items if not x[2]]), len(nonfire)))
print("训练集总图: %d" % len(sel))
print("训练集 label 文件: %d" % len(glob.glob(os.path.join(LBL, "*.txt"))))
print("OUT=%s" % OUT)
print("DONE")
