# -*- coding: utf-8 -*-
"""
analyze_fnfp.py —— 漏检/误报画像分析 (只读官方1100, 不进训练, 符合铁约束)
用指定模型在官方1100上推理, 量化:
  FN(漏火): 有GT火但模型(conf>=0.03)没检出类0 -> 记录火框大小/图亮度/尺寸
  FP(误报): 无GT火但模型检出类0 -> 记录图亮度/被误检区域
输出: analysis/fn_report.json, analysis/fp_report.json, 并把样本图复制到 analysis/ 供查看
用法: python analyze_fnfp.py <model.pt>
"""
import os, sys, glob, json, time, shutil
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(r"E:/火情监测/火情监测")
IMG_DIR = ROOT / "dataset_root" / "images" / "train"
LBL_DIR = ROOT / "dataset_root" / "labels" / "train"
AN = ROOT / "distill_pipeline" / "analysis"
FN_DIR = AN / "missed_fires"
FP_DIR = AN / "false_alarms"
for d in (AN, FN_DIR, FP_DIR):
    d.mkdir(parents=True, exist_ok=True)
DEVICE = sys.argv[2] if len(sys.argv) > 2 else "cpu"  # 默认CPU, 避免与训练抢GPU
FIRE_CONF = 0.03  # 与官方评估一致的"有火"判定阈值

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def brightness(img_path):
    try:
        im = Image.open(img_path).convert("L")
        return float(np.asarray(im).mean())
    except Exception:
        return -1.0


def gt_fire_boxes(lbl_path):
    boxes = []
    if lbl_path.exists() and lbl_path.stat().st_size > 0:
        for line in open(lbl_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            p = line.split()
            if int(p[0]) == 0:  # 火焰类
                boxes.append((float(p[3]), float(p[4])))  # w,h (normalized)
    return boxes


def main():
    if len(sys.argv) < 2:
        print("usage: python analyze_fnfp.py <model.pt>"); sys.exit(1)
    mp = sys.argv[1]
    from ultralytics import YOLO
    model = YOLO(mp)
    t0 = time.time()
    imgs = glob.glob(str(IMG_DIR / "*.jpg")) + glob.glob(str(IMG_DIR / "*.png"))
    # 推理(取所有检测, 后用阈值判定)
    res = model.predict(source=str(IMG_DIR), imgsz=640, conf=0.01, iou=0.45,
                         device=DEVICE, workers=0, batch=16, stream=True,
                         verbose=False, half=True)
    preds = {}
    for r in res:
        name = os.path.splitext(os.path.basename(r.path))[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            preds[name] = []
            continue
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        xywh = boxes.xywhn.cpu().numpy() if hasattr(boxes, "xywhn") else None
        preds[name] = list(zip(cls, conf))

    fn_list, fp_list = [], []
    fn_small = fn_dim = fn_dark = 0
    for im in imgs:
        name = os.path.splitext(os.path.basename(im))[0]
        lbl = LBL_DIR / (name + ".txt")
        gfb = gt_fire_boxes(lbl)
        has_fire = len(gfb) > 0
        pr = preds.get(name, [])
        fire_detected = any(c == 0 and cf >= FIRE_CONF for (c, cf) in pr)
        if has_fire and not fire_detected:
            W, H = Image.open(im).size
            areas = [w * h for (w, h) in gfb]
            br = brightness(im)
            rec = {"name": name, "W": W, "H": H, "n_boxes": len(gfb),
                   "min_area": round(min(areas), 5), "max_area": round(max(areas), 5),
                   "brightness": round(br, 1)}
            fn_list.append(rec)
            if min(areas) < 0.005:
                fn_small += 1
            if br < 60:
                fn_dark += 1
        elif (not has_fire) and fire_detected:
            W, H = Image.open(im).size
            br = brightness(im)
            rec = {"name": name, "W": W, "H": H, "brightness": round(br, 1)}
            fp_list.append(rec)
            try:
                shutil.copy(im, FP_DIR / f"{name}.jpg")
            except Exception:
                pass

    # 复制漏火图到分析目录
    for rec in fn_list:
        try:
            shutil.copy(IMG_DIR / f"{rec['name']}.jpg", FN_DIR / f"{rec['name']}.jpg")
        except Exception:
            # 尝试png
            try:
                shutil.copy(IMG_DIR / f"{rec['name']}.png", FN_DIR / f"{rec['name']}.png")
            except Exception:
                pass

    json.dump(fn_list, open(AN / "fn_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(fp_list, open(AN / "fp_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    n_fn = len(fn_list)
    n_fp = len(fp_list)
    print(f"[分析] 用时 {time.time()-t0:.0f}s", flush=True)
    print(f"[FN] 漏火图={n_fn}  其中小框(<0.5%面积)={fn_small}  偏暗(<60亮度)={fn_dark}", flush=True)
    if fn_list:
        import statistics as st
        ma = [r["min_area"] for r in fn_list]
        br = [r["brightness"] for r in fn_list if r["brightness"] >= 0]
        print(f"      漏火最小框面积: 中位={st.median(ma):.4f} 均值={st.mean(ma):.4f} 最小={min(ma):.4f}", flush=True)
        if br:
            print(f"      漏火图亮度: 中位={st.median(br):.1f} 均值={st.mean(br):.1f}", flush=True)
    print(f"[FP] 误报图={n_fp}", flush=True)
    if fp_list:
        import statistics as st
        br = [r["brightness"] for r in fp_list if r["brightness"] >= 0]
        if br:
            print(f"      误报图亮度: 中位={st.median(br):.1f} 均值={st.mean(br):.1f}", flush=True)
    print(f"[输出] {FN_DIR} ({n_fn}张)  {FP_DIR} ({n_fp}张)  reports in {AN}", flush=True)


if __name__ == "__main__":
    main()
