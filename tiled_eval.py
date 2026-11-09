# -*- coding: utf-8 -*-
"""
tiled_eval.py —— 在官方完整1100张上做图像级火焰 P/R，但用【切片(Tiled)+多尺度】推理。
针对小火(<1%面积)漏检死穴: 把大图切成重叠patch分别检测, 小火在patch内占比放大, 召回提升。
复用 eval_official_clean.py 的 GT 口径(只看类0)与 conf 扫描, 保证可比。
用法: python tiled_eval.py <model.pt> [PATCH] [OVERLAP]
"""
import os, sys, glob, time
from pathlib import Path

for _k, _v in {
    "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4", "VECLIB_MAXIMUM_THREADS": "4",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}.items():
    os.environ.setdefault(_k, _v)

import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path(r"E:/火情监测/火情监测")
IMG_DIR = ROOT / "dataset_root" / "images" / "train"
LBL_DIR = ROOT / "dataset_root" / "labels" / "train"
DEVICE = 0
CONF_SWEEP = [round(x * 0.01, 2) for x in range(3, 71)]    # 0.03 .. 0.70
WHOLE_IMGSZ = 1280
PATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 640
OVERLAP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
STRIDE = int(PATCH * (1 - OVERLAP))
INFER_CONF = 0.005   # 切片内部低阈值, 由外层 conf 扫描决定最终工作点


def build_gt():
    gt = {}
    imgs = glob.glob(str(IMG_DIR / "*.jpg")) + glob.glob(str(IMG_DIR / "*.png"))
    for im in imgs:
        name = os.path.splitext(os.path.basename(im))[0]
        lbl = LBL_DIR / (name + ".txt")
        has = False
        if lbl.exists() and lbl.stat().st_size > 0:
            for line in open(lbl, encoding="utf-8"):
                line = line.strip()
                if line and int(line.split()[0]) == 0:
                    has = True
                    break
        gt[name] = has
    return gt, len(imgs)


def infer_tiled(model_path):
    model = YOLO(str(model_path))
    out = {}
    imgs = glob.glob(str(IMG_DIR / "*.jpg")) + glob.glob(str(IMG_DIR / "*.png"))
    n = len(imgs)
    for idx, im in enumerate(imgs):
        name = os.path.splitext(os.path.basename(im))[0]
        img = Image.open(im).convert("RGB")
        W, H = img.size
        dets = []  # (cls, conf)

        # 1) 整图多尺度 (捕捉中/大火, 也作为补充)
        try:
            r_whole = model.predict(source=im, imgsz=WHOLE_IMGSZ, conf=INFER_CONF,
                                    iou=0.45, device=DEVICE, workers=0, batch=1,
                                    verbose=False, half=True, stream=False)
            b = r_whole[0].boxes
            if b is not None and len(b) > 0:
                cls = b.cls.cpu().numpy().astype(int)
                conf = b.conf.cpu().numpy()
                for c, cf in zip(cls, conf):
                    dets.append((int(c), float(cf)))
        except Exception as e:
            print(f"[whole] {name} failed {e}", flush=True)

        # 2) 切片 (放大小火)
        ys = list(range(0, max(1, H - PATCH + 1), STRIDE))
        xs = list(range(0, max(1, W - PATCH + 1), STRIDE))
        if H > PATCH:
            ys.append(H - PATCH)
        if W > PATCH:
            xs.append(W - PATCH)
        if not xs:
            xs = [0]
        if not ys:
            ys = [0]
        patches = []  # (x0, y0, pil)
        for y0 in ys:
            for x0 in xs:
                x1 = min(x0 + PATCH, W)
                y1 = min(y0 + PATCH, H)
                patch = img.crop((x0, y0, x1, y1))
                patches.append((x0, y0, patch))
        if patches:
            try:
                res = model.predict(source=[p[2] for p in patches], imgsz=PATCH,
                                    conf=INFER_CONF, iou=0.45, device=DEVICE,
                                    workers=0, batch=16, verbose=False, half=True,
                                    stream=False)
                for (x0, y0, _), r in zip(patches, res):
                    b = r.boxes
                    if b is None or len(b) == 0:
                        continue
                    cls = b.cls.cpu().numpy().astype(int)
                    conf = b.conf.cpu().numpy()
                    for c, cf in zip(cls, conf):
                        dets.append((int(c), float(cf)))
            except Exception as e:
                print(f"[tile] {name} failed {e}", flush=True)

        out[name] = dets
        if (idx + 1) % 100 == 0:
            print(f"[tiled] {idx+1}/{n} done", flush=True)
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: python tiled_eval.py <model.pt> [PATCH] [OVERLAP]")
        sys.exit(1)
    model_path = sys.argv[1]
    t0 = time.time()
    gt, n = build_gt()
    fire = sum(gt.values())
    print(f"[GT] 官方图={n} 有火={fire} 无火={n-fire} (无泄漏)", flush=True)
    print(f"[tiled] PATCH={PATCH} OVERLAP={OVERLAP} WHOLE={WHOLE_IMGSZ}", flush=True)

    preds = infer_tiled(model_path)
    print(f"\n===== {os.path.basename(model_path)} (TILED, 官方1100/无泄漏) =====", flush=True)
    print(f"{'conf':>5}  {'P':>7}  {'R':>7}  {'F1':>7}  {'TP':>5} {'FP':>5} {'FN':>5}", flush=True)
    best = None
    best_mm = None
    for t in CONF_SWEEP:
        tp = fp = fn = 0
        for name, has in gt.items():
            pr = preds.get(name, [])
            pf = any(c == 0 and cf >= t for (c, cf) in pr)
            if has and pf:
                tp += 1
            elif (not has) and pf:
                fp += 1
            elif has and (not pf):
                fn += 1
        P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        F = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0
        mn = min(P, R)
        print(f"{t:>5.2f}  {P:>7.4f}  {R:>7.4f}  {F:>7.4f}  {tp:>5d} {fp:>5d} {fn:>5d}", flush=True)
        if best is None or F > best[1]:
            best = (t, F, P, R, tp, fp, fn)
        if best_mm is None or mn > best_mm[0]:
            best_mm = (mn, t, P, R)
    tc, tF, tP, tR, tp, fp, fn = best
    print(f">>> BEST conf={tc:.2f} P={tP:.4f} R={tR:.4f} F1={tF:.4f} (TP={tp} FP={fp} FN={fn})", flush=True)
    print(f">>> MAXMIN conf={best_mm[1]:.2f} P={best_mm[2]:.4f} R={best_mm[3]:.4f} maxmin={best_mm[0]:.4f}", flush=True)
    print(f"[总用时] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
