# -*- coding: utf-8 -*-
"""
mine_targeted_pool.py —— 为 expF(针对性样本) / expG(蒸馏) 填池 (全部硬链接, 零磁盘占用)
- targeted_pool: 挖外部"小火"硬正样本(火框面积<SMALL_THR), 直击召回死穴(74%漏火是小框)
- distill_pseudo_pool: 放外部"非训练集"图(alirhayem valid/test + cctv + kien), 供教师伪标自训练
均只含外部数据, 绝不碰官方1100. 硬链接到源, 不复制.
用法: python mine_targeted_pool.py
"""
import os, glob, random
from pathlib import Path

ROOT = Path(r"E:/火情监测/火情监测")
EXT = ROOT / "external_datasets"
DP = ROOT / "distill_pipeline"
TI = DP / "targeted_pool" / "images"; TL = DP / "targeted_pool" / "labels"
PI = DP / "distill_pseudo_pool" / "images"; PL = DP / "distill_pseudo_pool" / "labels"
for d in (TI, TL, PI, PL):
    d.mkdir(parents=True, exist_ok=True)

SMALL_THR = 0.01       # 火框面积<1% 视为"小火"(漏检主因)
CAP_TARGETED = 25000
CAP_PSEUDO = 15000

# 训练源(挖小火): 名 -> (img_dir, lbl_dir)
TRAIN_SRC = {
    "alir": (EXT / "Kaggle_alirhayem_FireSmoke/Enhanced/Enhanced/train/images",
             EXT / "Kaggle_alirhayem_FireSmoke/Enhanced/Enhanced/train/labels"),
    "dfire": (EXT / "academic_D-Fire/train/images",
              EXT / "academic_D-Fire/train/labels"),
    "cctv": (EXT / "HF_CCTV_Smoke_Fire/CCTV_Fire_Smoke_Emergency_Detection_Dataset/images",
             EXT / "HF_CCTV_Smoke_Fire/CCTV_Fire_Smoke_Emergency_Detection_Dataset/labels"),
    "kien": (EXT / "HF_KienNgyuen_FireSmoke/_unzipped/CCTV Smoke & Fire Emergency Detection Dataset/images",
             EXT / "HF_KienNgyuen_FireSmoke/_unzipped/CCTV Smoke & Fire Emergency Detection Dataset/labels"),
}
# 伪标源(非训练集图): 名 -> img_dir (不限标签, 教师会重新伪标)
PSEUDO_SRC = {
    "alirval": EXT / "Kaggle_alirhayem_FireSmoke/Enhanced/Enhanced/valid/images",
    "alirtest": EXT / "Kaggle_alirhayem_FireSmoke/Enhanced/Enhanced/test/images",
    "cctv": EXT / "HF_CCTV_Smoke_Fire/CCTV_Fire_Smoke_Emergency_Detection_Dataset/images",
    "kien": EXT / "HF_KienNgyuen_FireSmoke/_unzipped/CCTV Smoke & Fire Emergency Detection Dataset/images",
}


def hardlink(src, dst):
    if dst.exists():
        return True
    try:
        os.link(src, dst)
        return True
    except OSError:
        try:
            if not dst.exists():
                import shutil
                shutil.copy(src, dst)
            return True
        except Exception:
            return False


def mine_small_fire():
    picked = []
    for name, (img_dir, lbl_dir) in TRAIN_SRC.items():
        if not lbl_dir.exists():
            continue
        labs = sorted(glob.glob(str(lbl_dir / "*.txt")))
        for lp in labs:
            stem = os.path.splitext(os.path.basename(lp))[0]
            # 解析类0火框, 取最小面积
            min_area = None
            try:
                for line in open(lp, encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    p = line.split()
                    if int(p[0]) == 0:
                        a = float(p[3]) * float(p[4])
                        min_area = a if min_area is None else min(min_area, a)
            except Exception:
                continue
            if min_area is None or min_area >= SMALL_THR:
                continue
            im = None
            for ext in (".jpg", ".JPG", ".png", ".PNG"):
                c = img_dir / (stem + ext)
                if c.exists():
                    im = c; break
            if im is None:
                continue
            picked.append((name, stem, im, Path(lp)))
    # 打乱后截取上限
    random.seed(7); random.shuffle(picked)
    if len(picked) > CAP_TARGETED:
        picked = picked[:CAP_TARGETED]
    n = 0
    for name, stem, im, lp in picked:
        newstem = f"{name}_{stem}"
        if hardlink(im, TI / (newstem + im.suffix)) and hardlink(lp, TL / (newstem + ".txt")):
            n += 1
    print(f"[targeted] 挖到小火硬正样本 {n} 张 (阈值面积<{SMALL_THR})", flush=True)
    return n


def mine_pseudo():
    picked = []
    for name, img_dir in PSEUDO_SRC.items():
        if not img_dir.exists():
            continue
        for ext in (".jpg", ".JPG", ".png", ".PNG"):
            for ip in glob.glob(str(img_dir / f"*{ext}")):
                picked.append((name, ip))
    random.seed(11); random.shuffle(picked)
    if len(picked) > CAP_PSEUDO:
        picked = picked[:CAP_PSEUDO]
    n = 0
    for name, ip in picked:
        stem = os.path.splitext(os.path.basename(ip))[0]
        newstem = f"{name}_{stem}"
        if hardlink(Path(ip), PI / (newstem + Path(ip).suffix)):
            n += 1
    print(f"[pseudo] 填蒸馏伪标池 {n} 张 (外部非训练集图)", flush=True)
    return n


if __name__ == "__main__":
    t = mine_small_fire()
    p = mine_pseudo()
    print(f"[done] targeted={t}  pseudo={p}", flush=True)
