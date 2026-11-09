# -*- coding: utf-8 -*-
"""
eval_official_clean.py —— 在【官方完整 1100 张】上做图像级火焰 P/R（纯官方标注当GT）。
这是用户定义的"永远的测试集"：官方图一张都不进训练，故本评估无泄漏、可信。
用法: python eval_official_clean.py <model.pt>
"""
import os, sys, glob, time
from pathlib import Path

for _k, _v in {
    "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4", "VECLIB_MAXIMUM_THREADS": "4",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}.items():
    os.environ.setdefault(_k, _v)

from ultralytics import YOLO

ROOT = Path(r"E:/火情监测/火情监测")
IMG_DIR = ROOT / "dataset_root" / "images" / "train"      # 官方完整图 (1088张)
LBL_DIR = ROOT / "dataset_root" / "labels" / "train"      # 纯官方GT
DEVICE = 0
CONF_SWEEP = [round(x * 0.01, 2) for x in range(3, 71)]    # 0.03 .. 0.70 (细扫)


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
                if line and int(line.split()[0]) == 0:   # 只看火焰类0
                    has = True
                    break
        gt[name] = has
    return gt, len(imgs)


def infer(model_path):
    model = YOLO(str(model_path))
    try:
        res = model.predict(source=str(IMG_DIR), imgsz=640, conf=0.01, iou=0.45,
                             device=DEVICE, workers=0, batch=16, stream=True,
                             verbose=False, half=True)
    except Exception as e:
        print(f"[infer] half failed {e}, retry half=False", flush=True)
        res = model.predict(source=str(IMG_DIR), imgsz=640, conf=0.01, iou=0.45,
                             device=DEVICE, workers=0, batch=16, stream=True,
                             verbose=False)
    out = {}
    for r in res:
        name = os.path.splitext(os.path.basename(r.path))[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            out[name] = []
            continue
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        out[name] = list(zip(cls, conf))
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: python eval_official_clean.py <model.pt>")
        sys.exit(1)
    model_path = sys.argv[1]
    t0 = time.time()
    gt, n = build_gt()
    fire = sum(gt.values())
    print(f"[GT] 官方图={n} 有火={fire} 无火={n-fire} (纯官方标注, 题目1100张口径, 无泄漏)", flush=True)

    preds = infer(model_path)
    print(f"\n===== {os.path.basename(model_path)} (官方1100/无泄漏) =====", flush=True)
    print(f"{'conf':>5}  {'P':>7}  {'R':>7}  {'F1':>7}  {'TP':>5} {'FP':>5} {'FN':>5}", flush=True)
    best = None
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
        flag = " *" if t in (0.05, 0.10) else ""
        print(f"{t:>5.2f}  {P:>7.4f}  {R:>7.4f}  {F:>7.4f}  {tp:>5d} {fp:>5d} {fn:>5d}{flag}", flush=True)
        if best is None or F > best[1]:
            best = (t, F, P, R, tp, fp, fn)
    tc, tF, tP, tR, tp, fp, fn = best
    print(f">>> BEST conf={tc:.2f} P={tP:.4f} R={tR:.4f} F1={tF:.4f} (TP={tp} FP={fp} FN={fn})", flush=True)
    print(f"[总用时] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
