# -*- coding: utf-8 -*-
"""
eval_clean_gt.py —— 用【纯官方标注】当 Ground Truth，对蒸馏验证集(官方图)重算图像级火焰 P/R。
目的: 验证之前 0.93/0.90 的评估是否受"教师伪标签混入验证标签"影响。
对比: eval_only.py 用 distill_dataset/labels/val(官方+教师伪标签) 当GT；
      本脚本用 dataset_root/labels/train(纯官方) 当GT。图像同一批(官方1100里的109张)。
结论预期: 两者 has_fire 判定一致(教师伪标签只追加在正样本图, 不改变图像级真值) → 分数应基本一致。
"""
import os, glob, time
from pathlib import Path

for _k, _v in {
    "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4", "VECLIB_MAXIMUM_THREADS": "4",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}.items():
    os.environ.setdefault(_k, _v)

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

ROOT = Path(r"E:/火情监测/火情监测")
VAL_IMG = ROOT / "distill_pipeline" / "distill_dataset" / "images" / "val"
OFF_LBL = ROOT / "dataset_root" / "labels" / "train"   # 纯官方标注
MODEL = ROOT / "distill_pipeline" / "final_yolo26n_best.pt"
DEVICE = 0
CONF_SWEEP = [round(x * 0.05, 2) for x in range(2, 19)]  # 0.10 .. 0.90


def main():
    t0 = time.time()
    # ---- GT: 仅官方标注 ----
    gt = {}
    imgs = glob.glob(str(VAL_IMG / "*.jpg")) + glob.glob(str(VAL_IMG / "*.png"))
    for im in imgs:
        name = os.path.splitext(os.path.basename(im))[0]
        lbl = OFF_LBL / (name + ".txt")
        has = False
        if lbl.exists() and lbl.stat().st_size > 0:
            for line in open(lbl, encoding="utf-8"):
                line = line.strip()
                if line and int(line.split()[0]) == 0:
                    has = True
                    break
        gt[name] = has
    print(f"[clean-GT] val imgs={len(gt)} fire={sum(gt.values())} "
          f"nofire={len(gt)-sum(gt.values())}", flush=True)

    # ---- 预测 ----
    model = YOLO(str(MODEL))
    try:
        results = model.predict(source=str(VAL_IMG), imgsz=640, conf=0.01, iou=0.45,
                                device=DEVICE, workers=0, batch=8, stream=True,
                                verbose=False, half=True)
    except Exception as e:
        print(f"[clean-GT] half failed {e}, retry half=False", flush=True)
        results = model.predict(source=str(VAL_IMG), imgsz=640, conf=0.01, iou=0.45,
                                device=DEVICE, workers=0, batch=8, stream=True,
                                verbose=False)
    boxes_by_img = {}
    for r in results:
        name = os.path.splitext(os.path.basename(r.path))[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            boxes_by_img[name] = []
            continue
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        boxes_by_img[name] = list(zip(cls, conf))

    # ---- 阈值扫描 ----
    best = None
    print("[clean-GT] conf      P       R       F1      TP  FP  FN", flush=True)
    for t in CONF_SWEEP:
        tp = fp = fn = 0
        for name, has in gt.items():
            preds = boxes_by_img.get(name, [])
            pred_fire = any(c == 0 and cf >= t for (c, cf) in preds)
            if has and pred_fire:
                tp += 1
            elif (not has) and pred_fire:
                fp += 1
            elif has and (not pred_fire):
                fn += 1
        P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        F = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0
        print(f"           {t:.2f}  {P:.4f}  {R:.4f}  {F:.4f}  {tp:3d} {fp:3d} {fn:3d}", flush=True)
        if best is None or F > best[1]:
            best = (t, F, P, R, tp, fp, fn)
    tc, tF, tP, tR, _, _, _ = best
    print(f"[clean-GT] BEST conf={tc:.2f} P={tP:.4f} R={tR:.4f} F1={tF:.4f} "
          f"(用时 {time.time()-t0:.0f}s)", flush=True)
    print(f"[clean-GT] 对比旧评估(eval_only, 混伪标签GT): conf=0.10 P=0.9318 R=0.9011 F1=0.9162", flush=True)


if __name__ == "__main__":
    main()
