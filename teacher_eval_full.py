# -*- coding: utf-8 -*-
"""
teacher_eval_full.py
用【纯官方标注】当 Ground Truth，对官方完整训练集图(1088张≈题目1100张)做图像级火焰 P/R。
跑两个模型对比：
  - 教师 YOLOv11s (firedetect-11s.pt)  —— 未在我们数据上训练, 无泄漏, 代表"继承上界"
  - 学生 yolo26n (final_yolo26n_best.pt) —— 注: 学生在此991张上训练过, 含训练泄漏, 仅作参照
目的:
  1) 测教师上界, 决策蒸馏是否值得;
  2) 回应"用题目给的1100张而非子集" —— 这里用的是官方完整图, 不是109张子集。
图像级口径: 只看"火焰(类0)"框 —— 图里有任意类0框且conf>=t → 预测有火(1); 否则0。
官方GT同理: 图标注里存在类0行 → 真值有火(1); 否则0 (烟雾类1框不算有火)。
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
IMG_DIR = ROOT / "dataset_root" / "images" / "train"      # 官方完整图 (1088张)
LBL_DIR = ROOT / "dataset_root" / "labels" / "train"      # 纯官方GT
TEACHER = ROOT / "external_datasets" / "teachers" / "leeyunjai_yolo11_firedetect" / "firedetect-11s.pt"
STUDENT = ROOT / "distill_pipeline" / "final_yolo26n_best.pt"
DEVICE = 0
CONF_SWEEP = [round(x * 0.05, 2) for x in range(1, 19)]   # 0.05 .. 0.90


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


def sweep(gt, preds, tag):
    print(f"\n===== {tag} =====", flush=True)
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
        print(f"{t:>5.2f}  {P:>7.4f}  {R:>7.4f}  {F:>7.4f}  {tp:>5d} {fp:>5d} {fn:>5d}", flush=True)
        if best is None or F > best[1]:
            best = (t, F, P, R)
    tc, tF, tP, tR = best
    print(f">>> {tag} BEST conf={tc:.2f} P={tP:.4f} R={tR:.4f} F1={tF:.4f}", flush=True)
    return best


def main():
    t0 = time.time()
    gt, n = build_gt()
    print(f"[GT] 官方图={n} 有火={sum(gt.values())} 无火={n - sum(gt.values())} (纯官方标注, 题目1100张口径)", flush=True)
    print(f"[注意] 学生在此{n}张中的991张训练过, 其结果含训练泄漏, 仅作参照; 教师无泄漏, 代表继承上界。", flush=True)

    tpred = infer(TEACHER)
    sweep(gt, tpred, "教师 YOLOv11s (firedetect-11s) — 1100张口径/无泄漏")

    spred = infer(STUDENT)
    sweep(gt, spred, "学生 yolo26n (final) — 1100张口径/含训练泄漏(参照)")

    print(f"\n[总用时] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
