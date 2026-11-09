# -*- coding: utf-8 -*-
"""
conf_sweep_student.py
对【独立109张验证集】(纯官方GT, 无泄漏) 扫描学生 yolo26n 的图像级火焰 P/R 各阈值。
目的: 给出交卷选 conf 工作点的干净推荐表(竞赛真指标基准, 不含训练泄漏)。
阈值覆盖: 低阈值(偏向不漏火) → 高阈值(偏向不误报)。
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
VAL_IMG = ROOT / "distill_pipeline" / "distill_dataset" / "images" / "val"  # 独立109集
OFF_LBL = ROOT / "dataset_root" / "labels" / "train"                       # 纯官方GT
MODEL = ROOT / "distill_pipeline" / "final_yolo26n_best.pt"
DEVICE = 0
# 更细的阈值, 覆盖低阈值(漏火区)和高阈值(误报区)
CONF_SWEEP = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70]


def main():
    t0 = time.time()
    gt = {}
    imgs = glob.glob(str(VAL_IMG / "*.jpg")) + glob.glob(str(VAL_IMG / "*.png"))
    for im in imgs:
        name = os.path.splitext(os.path.basename(im))[0]
        lbl = OFF_LBL / (name + ".txt")
        has = False
        if lbl.exists() and lbl.stat().st_size > 0:
            for line in open(lbl, encoding="utf-8"):
                line = line.strip()
                if line and int(line.split()[0]) == 0:   # 只看火焰类0
                    has = True
                    break
        gt[name] = has
    print(f"[GT] 独立验证集={len(gt)} 有火={sum(gt.values())} 无火={len(gt) - sum(gt.values())} (纯官方标注/无泄漏)", flush=True)

    model = YOLO(str(MODEL))
    try:
        res = model.predict(source=str(VAL_IMG), imgsz=640, conf=0.001, iou=0.45,
                             device=DEVICE, workers=0, batch=8, stream=True,
                             verbose=False, half=True)
    except Exception as e:
        print(f"[warn] half failed {e}, retry", flush=True)
        res = model.predict(source=str(VAL_IMG), imgsz=640, conf=0.001, iou=0.45,
                             device=DEVICE, workers=0, batch=8, stream=True, verbose=False)
    preds = {}
    for r in res:
        name = os.path.splitext(os.path.basename(r.path))[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            preds[name] = []
            continue
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        preds[name] = list(zip(cls, conf))

    print(f"\n{'conf':>5}  {'P':>7}  {'R':>7}  {'F1':>7}  {'TP':>4} {'FP':>4} {'FN':>4}", flush=True)
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
        print(f"{t:>5.2f}  {P:>7.4f}  {R:>7.4f}  {F:>7.4f}  {tp:>4d} {fp:>4d} {fn:>4d}", flush=True)
        if best is None or F > best[1]:
            best = (t, F, P, R)
    tc, tF, tP, tR = best
    print(f"\n>>> 推荐提交工作点(无泄漏): conf={tc:.2f} P={tP:.4f} R={tR:.4f} F1={tF:.4f} (用时{time.time()-t0:.0f}s)", flush=True)
    print(f">>> 用法: python infer_image_level.py --model final_yolo26n_best.pt --source <测试图目录> --conf {tc:.2f} --out result.json", flush=True)


if __name__ == "__main__":
    main()
