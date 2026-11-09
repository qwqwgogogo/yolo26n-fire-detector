# -*- coding: utf-8 -*-
"""
fused_eval.py —— 在官方完整1100张上做图像级火焰 P/R，用【切片 + 整图一致性门控】推理。
在 tiled_eval 基础上解决精度短板: 切片把召回拉满, 但孤立暖斑(小块里的伪火)带来~200 FP。
门控规则: 一个切片类0框只有在【与整图(640,低阈值)火框重叠】或【该图切片火框数>=2(多patch一致)】
时才算"supported"; 孤立单patch框直接丢弃 -> 压FP不伤召回。零训练成本。
用法: python fused_eval.py <model.pt> [PATCH] [OVERLAP]
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
WHOLE_IMGSZ = 1280          # 多尺度整图(捕捉中大火)
GATE_IMGSZ = 640            # 整图门控(低阈值, 抓小火微弱响应)
GATE_CONF = 0.001
PATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 640
OVERLAP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
STRIDE = int(PATCH * (1 - OVERLAP))
INFER_CONF = 0.005


def iou(a, b):
    # a,b: [x0,y0,x1,y1] 归一化
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


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


def infer_fused(model_path):
    model = YOLO(str(model_path))
    out = {}
    imgs = glob.glob(str(IMG_DIR / "*.jpg")) + glob.glob(str(IMG_DIR / "*.png"))
    n = len(imgs)
    for idx, im in enumerate(imgs):
        name = os.path.splitext(os.path.basename(im))[0]
        img = Image.open(im).convert("RGB")
        W, H = img.size

        # 1) 整图门控框(低阈值, 所有类0框作为"火可能存在区域"); 同时多尺度整图补充
        gate_boxes = []   # 全局归一化 xyxy
        try:
            for sz in (GATE_IMGSZ, WHOLE_IMGSZ):
                r = model.predict(source=im, imgsz=sz, conf=GATE_CONF,
                                  iou=0.9, device=DEVICE, workers=0, batch=1,
                                  verbose=False, half=True, stream=False)
                b = r[0].boxes
                if b is not None and len(b) > 0:
                    cls = b.cls.cpu().numpy().astype(int)
                    xy = b.xyxyn.cpu().numpy()
                    cf = b.conf.cpu().numpy()
                    for c, box, cfc in zip(cls, xy, cf):
                        if int(c) == 0:
                            gate_boxes.append((box.tolist(), float(cfc)))
        except Exception as e:
            print(f"[gate] {name} failed {e}", flush=True)

        # 2) 切片
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
        patches = []
        for y0 in ys:
            for x0 in xs:
                x1 = min(x0 + PATCH, W)
                y1 = min(y0 + PATCH, H)
                patches.append((x0, y0, img.crop((x0, y0, x1, y1))))
        tile_boxes = []  # (global xyxy norm, conf)
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
                    xy = b.xyxyn.cpu().numpy()
                    for c, cfc, box in zip(cls, conf, xy):
                        if int(c) != 0:
                            continue
                        xc, yc, wc, hc = box
                        gx0 = (x0 + xc * PATCH) / W
                        gy0 = (y0 + yc * PATCH) / H
                        gx1 = (x0 + (xc + wc) * PATCH) / W
                        gy1 = (y0 + (yc + hc) * PATCH) / H
                        tile_boxes.append(([gx0, gy0, gx1, gy1], float(cfc)))
            except Exception as e:
                print(f"[tile] {name} failed {e}", flush=True)

        n_tile = len(tile_boxes)
        sup_conf = []
        for (gbox, cfc) in tile_boxes:
            supp = (n_tile >= 2)
            if not supp:
                for (wb, _) in gate_boxes:
                    if iou(gbox, wb) > 0.05:
                        supp = True
                        break
            if supp:
                sup_conf.append(cfc)
        gate_conf = [cfc for (_, cfc) in gate_boxes]
        out[name] = {"sup": sup_conf, "gate": gate_conf}
        if (idx + 1) % 100 == 0:
            print(f"[fused] {idx+1}/{n} done", flush=True)
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: python fused_eval.py <model.pt> [PATCH] [OVERLAP]")
        sys.exit(1)
    model_path = sys.argv[1]
    t0 = time.time()
    gt, n = build_gt()
    fire = sum(gt.values())
    print(f"[GT] 官方图={n} 有火={fire} 无火={n-fire} (无泄漏)", flush=True)
    print(f"[fused] PATCH={PATCH} OVERLAP={OVERLAP} GATE={GATE_IMGSZ}@{GATE_CONF}", flush=True)
    preds = infer_fused(model_path)
    print(f"\n===== {os.path.basename(model_path)} (FUSED, 官方1100/无泄漏) =====", flush=True)
    print(f"{'conf':>5}  {'P':>7}  {'R':>7}  {'F1':>7}  {'TP':>5} {'FP':>5} {'FN':>5}", flush=True)
    best = None
    best_mm = None
    for t in CONF_SWEEP:
        tp = fp = fn = 0
        for name, has in gt.items():
            d = preds.get(name, {"sup": [], "gate": []})
            pf = any(c >= t for c in d["sup"]) or any(c >= t for c in d["gate"])
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
