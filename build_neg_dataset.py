# -*- coding: utf-8 -*-
"""
从本地已下载的 D-Fire 公开集挖掘"阴性/假火"样本，与官方 1100 合并成训练集。
阴性定义：D-Fire 标注中不含 class0(火) 的图 = 纯烟图 + 空背景图。
这些图视觉上常含红色/橙色/灯光/烟雾，正好是压"假火"误报的弹药。
官方 1100（含火框标注）作为正样本主体并入。
硬链接省空间（同盘 E:，不复制大图）。
"""
import os, glob, random, shutil

random.seed(42)
SRC = "E:/火情监测/火情监测/external_datasets/academic_D-Fire"
OFF_IMG = "E:/火情监测/火情监测/dataset_root/images/train"
OFF_LBL = "E:/火情监测/火情监测/dataset_root/labels/train"
OUT = "E:/火情监测/火情监测/distill_pipeline/neg_fire_dataset"
IMG = os.path.join(OUT, "images", "train")
LBL = os.path.join(OUT, "labels", "train")
BG_KEEP = 5000  # 背景图抽样上限，控制规模避免训练过久

os.makedirs(IMG, exist_ok=True)
os.makedirs(LBL, exist_ok=True)

def link(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy(src, dst)

# ---- 1. 扫 D-Fire，筛阴性（无 class0）----
neg = []  # (img_path, kind)
for lp in glob.glob(os.path.join(SRC, "**", "labels", "*.txt"), recursive=True):
    base = os.path.splitext(os.path.basename(lp))[0]
    img = None
    for ext in (".jpg", ".png", ".jpeg"):
        cand = os.path.join(os.path.dirname(os.path.dirname(lp)), "images", base + ext)
        if os.path.exists(cand):
            img = cand
            break
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
    if not has_fire:
        kind = "smoke" if os.path.getsize(lp) > 0 else "bg"
        neg.append((img, kind))

smoke = [x for x in neg if x[1] == "smoke"]
bg = [x for x in neg if x[1] == "bg"]
random.shuffle(bg)
bg = bg[:BG_KEEP]
sel = smoke + bg

for img, _ in sel:
    link(img, os.path.join(IMG, os.path.basename(img)))

# ---- 2. 官方 1100 并入（图+label）----
n_off = 0
for fp in glob.glob(os.path.join(OFF_IMG, "*.jpg")):
    base = os.path.splitext(os.path.basename(fp))[0]
    link(fp, os.path.join(IMG, os.path.basename(fp)))
    lbl = os.path.join(OFF_LBL, base + ".txt")
    if os.path.exists(lbl):
        d2 = os.path.join(LBL, base + ".txt")
        if not os.path.exists(d2):
            shutil.copy(lbl, d2)
    n_off += 1

# ---- 3. yaml ----
with open(os.path.join(OUT, "dataset.yaml"), "w", encoding="utf-8") as f:
    f.write("path: %s\n" % OUT)
    f.write("train: images/train\n")
    f.write("val: images/train\n")   # 评估走独立脚本，不用此 val
    f.write("names: {0: fire, 1: smoke}\n")

print("D-Fire 阴性: smoke=%d bg=%d(抽样%d) total=%d" % (len(smoke), len([x for x in neg if x[1]=='bg']), len(bg), len(sel)))
print("官方并入: %d 图" % n_off)
print("训练集总图: %d" % (len(sel) + n_off))
print("训练集 label 文件: %d" % len(glob.glob(os.path.join(LBL, "*.txt"))))
print("OUT=%s" % OUT)
print("DONE")
