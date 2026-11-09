# -*- coding: utf-8 -*-
"""
gen_pseudo.py  —— 离线知识蒸馏第 1 步：用 YOLOv11 教师模型生成伪标签并与真实标注合并。

教师: external_datasets/teachers/leeyunjai_yolo11_firedetect/firedetect-11x.pt (失败回退 11s)
  - 类别与数据集一致: 0=Fire(火), 1=Smoke(烟)
  - 仅对"有真实标注的正样本图"补充教师检测到、且与 GT 不重叠的高置信框，
    以扩充训练信号、提升召回；负样本图保持为空(不注入噪声)。
学生: YOLOv26 (在 run_distill.py 中训练)
输出: distill_pipeline/distill_dataset/{images,labels}/{train,val} + dataset.yaml
"""
import os, sys, shutil, glob, random, math
from pathlib import Path

# 线程限制(防御性): 避免小页面文件下 import torch 失败
for _k, _v in {
    "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4", "VECLIB_MAXIMUM_THREADS": "4",
}.items():
    os.environ.setdefault(_k, _v)

ROOT = Path(r"E:/火情监测/火情监测")
SRC_IMG = ROOT / "dataset_root" / "images" / "train"
SRC_LBL = ROOT / "dataset_root" / "labels" / "train"
OUT = ROOT / "distill_pipeline" / "distill_dataset"
TEACHER_X = ROOT / "external_datasets" / "teachers" / "leeyunjai_yolo11_firedetect" / "firedetect-11x.pt"
TEACHER_S = ROOT / "external_datasets" / "teachers" / "leeyunjai_yolo11_firedetect" / "firedetect-11s.pt"
MARKER = OUT / ".pseudo_done"

T_TEACHER = 0.25   # 教师推理时保留的置信阈值(低, 多召回)
T_ADD = 0.45       # 仅当教师预测 conf>=此值 才作为伪标签加入
IOU_KEEP = 0.30    # 与任一 GT 框 IoU < 此值 才视为"新增检测"
SEED = 42
VAL_RATIO = 0.10   # 验证集比例(分层: 正/负各自抽取)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = (ax2 - ax1) * (ay2 - ay1)
    bb = (bx2 - bx1) * (by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def link_or_copy(src, dst):
    try:
        if not dst.exists():
            os.link(src, dst)
            os.chmod(dst, 0o777)
    except Exception:
        if not dst.exists():
            shutil.copy(src, dst)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if MARKER.exists():
        print("[gen_pseudo] marker exists, skip.")
        return

    # ---- 载入教师(优先 11s: 稳定且对该任务足够强; 11x 仅作回退) ----
    teacher = None
    for tp in (TEACHER_S, TEACHER_X):
        if tp.exists():
            try:
                from ultralytics import YOLO
                teacher = YOLO(str(tp))
                print("[gen_pseudo] teacher loaded:", tp.name)
                break
            except Exception as e:
                print("[gen_pseudo] load fail", tp.name, repr(e))
    if teacher is None:
        raise RuntimeError("no teacher model found")

    # ---- 收集图像, 区分正/负样本 ----
    imgs = sorted(glob.glob(str(SRC_IMG / "*.jpg"))) + sorted(glob.glob(str(SRC_IMG / "*.png")))
    pos, neg = [], []
    for im in imgs:
        name = os.path.splitext(os.path.basename(im))[0]
        lbl = os.path.join(str(SRC_LBL), name + ".txt")
        ispos = os.path.exists(lbl) and os.path.getsize(lbl) > 0
        (pos if ispos else neg).append(im)
    print(f"[gen_pseudo] total={len(imgs)} pos={len(pos)} neg={len(neg)}")

    random.seed(SEED)
    random.shuffle(pos)
    random.shuffle(neg)
    nval_pos = max(1, int(len(pos) * VAL_RATIO))
    nval_neg = max(1, int(len(neg) * VAL_RATIO))
    val_pos, train_pos = pos[:nval_pos], pos[nval_pos:]
    val_neg, train_neg = neg[:nval_neg], neg[nval_neg:]
    split = {"train": train_pos + train_neg, "val": val_pos + val_neg}
    print(f"[gen_pseudo] train={len(split['train'])} val={len(split['val'])}")

    # ---- 教师推理(带 OOM 自动降级链) ----
    print("[gen_pseudo] running teacher prediction ...")
    results = None
    specs = [
        (True, 16, 8, 640),
        (False, 16, 8, 640),
        (True, 8, 4, 512),
        (False, 8, 4, 512),
        (True, 1, 2, 416),
    ]
    for (half, batch, workers, imgsz) in specs:
        try:
            results = teacher.predict(
                source=str(SRC_IMG), imgsz=imgsz, conf=T_TEACHER, iou=0.45,
                device=0, workers=workers, batch=batch, stream=True,
                verbose=False, half=half,
            )
            print(f"[gen_pseudo] predict OK half={half} batch={batch} imgsz={imgsz}")
            break
        except Exception as e:
            print(f"[gen_pseudo] predict err spec(half={half},batch={batch},imgsz={imgsz}):", repr(e))
    if results is None:
        raise RuntimeError("teacher predict failed")

    preds = {}
    for r in results:
        name = os.path.splitext(os.path.basename(r.path))[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            preds[name] = []
            continue
        xyxyn = boxes.xyxyn.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        arr = []
        for i in range(len(cls)):
            x1, y1, x2, y2 = xyxyn[i]
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            arr.append((int(cls[i]), float(conf[i]), (x1, y1, x2, y2), (cx, cy, w, h)))
        preds[name] = arr
    print(f"[gen_pseudo] teacher predicted {len(preds)} images")

    # ---- 写出数据集 ----
    for sp, imlist in split.items():
        imdir = OUT / "images" / sp
        lbdir = OUT / "labels" / sp
        imdir.mkdir(parents=True, exist_ok=True)
        lbdir.mkdir(parents=True, exist_ok=True)
        for im in imlist:
            name = os.path.splitext(os.path.basename(im))[0]
            dst = imdir / os.path.basename(im)
            link_or_copy(im, dst)

            lblsrc = os.path.join(str(SRC_LBL), name + ".txt")
            lines = []
            gt = []
            if os.path.exists(lblsrc) and os.path.getsize(lblsrc) > 0:
                with open(lblsrc, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        p = line.split()
                        c = int(p[0]); cx, cy, w, h = map(float, p[1:5])
                        lines.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                        gt.append((c, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
            # 仅对正样本图补充教师伪标签(不重叠 GT 的高置信框)
            if gt:
                for (c, conf, xyxy, xywh) in preds.get(name, []):
                    if conf < T_ADD:
                        continue
                    overlap = max([iou(xyxy, g) for (_, g) in gt], default=0.0)
                    if overlap < IOU_KEEP:
                        lines.append(f"{c} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}")
            with open(lbdir / (name + ".txt"), "w", encoding="utf-8") as f:
                f.write(("\n".join(lines) + "\n") if lines else "")

    # ---- dataset.yaml ----
    yaml_path = OUT / "dataset.yaml"
    yaml_path.write_text(
        "# Auto-generated distillation dataset\n"
        f"path: {str(OUT).replace(chr(92), '/')}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 2\n"
        "names: [火, 烟]\n",
        encoding="utf-8",
    )

    # ---- 统计 ----
    def count_inst(split_name):
        n_img = len(split[split_name])
        n_fire = n_smoke = 0
        for f in (lbdir_parent := (OUT / "labels" / split_name)).glob("*.txt"):
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                cid = int(line.split()[0])
                if cid == 0:
                    n_fire += 1
                elif cid == 1:
                    n_smoke += 1
        return n_img, n_fire, n_smoke

    tr = count_inst("train"); vl = count_inst("val")
    MARKER.write_text(f"train={tr} val={vl}\n", encoding="utf-8")
    print(f"[gen_pseudo] DONE. train imgs={tr[0]} fire={tr[1]} smoke={tr[2]} | "
          f"val imgs={vl[0]} fire={vl[1]} smoke={vl[2]}")


if __name__ == "__main__":
    main()
