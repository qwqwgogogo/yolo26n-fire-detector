# -*- coding: utf-8 -*-
"""
eval_only.py —— 跳过训练, 直接用已有 best.pt 做图像级火焰 P/R 阈值扫描 + 生成提交格式 result.json + RESULT.md。

用途: 当训练因系统内存耗尽无法继续时, 用已有部分训练的 best.pt 完成竞赛交付物。
竞赛评分 = 图像级 Precision/Recall (不是 mAP), 所以阈值调优比堆 epoch 更重要。
"""
import os, sys, json, glob, time, shutil
from pathlib import Path

ROOT = Path(r"E:/火情监测/火情监测")
PIPE = ROOT / "distill_pipeline"
OUT = PIPE / "distill_dataset"
RUNS = PIPE / "runs"
LOG = PIPE / "run_distill.log"

# 线程限制 (必须在 import torch 前)
for _k, _v in {
    "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
    "VECLIB_MAXIMUM_THREADS": "4",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}.items():
    os.environ.setdefault(_k, _v)

import torch
from ultralytics import YOLO

DEVICE = 0
CONF_SWEEP = [round(x * 0.05, 2) for x in range(2, 19)]  # 0.10 .. 0.90


def log(s):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + str(s)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def find_best_pt():
    """找 epoch 数最多的训练目录里的 best.pt。"""
    best_dir = None
    best_ep = -1
    for r in glob.glob(str(RUNS / "distill_yolo26n_*")):
        rc = Path(r) / "results.csv"
        if rc.exists():
            try:
                with open(rc, encoding="utf-8") as f:
                    n = sum(1 for _ in f) - 1
            except Exception:
                n = 0
            if n > best_ep:
                best_ep = n
                best_dir = r
    if best_dir:
        bp = Path(best_dir) / "weights" / "best.pt"
        if bp.exists():
            log(f"[find] best.pt from {best_dir} ({best_ep} epochs): {bp}")
            return str(bp), best_dir, best_ep
    # fallback: 直接搜
    for r in glob.glob(str(RUNS / "distill_yolo26n_*")):
        bp = Path(r) / "weights" / "best.pt"
        if bp.exists():
            log(f"[find] fallback best.pt: {bp}")
            return str(bp), r, 0
    return None, None, 0


def image_level_eval(model_path, confs):
    """图像级火焰 P/R 扫描: 仅类0(火焰)框 conf>=阈值 -> 图=1, 否则=0。"""
    val_img = OUT / "images" / "val"
    val_lbl = OUT / "labels" / "val"

    # GT: 图有火 = 任一标注类0
    gt = {}
    for lbl in glob.glob(str(val_lbl / "*.txt")):
        name = os.path.splitext(os.path.basename(lbl))[0]
        has_fire = False
        for line in open(lbl, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            if int(line.split()[0]) == 0:
                has_fire = True
                break
        gt[name] = has_fire

    # 补充无标注的图(负样本)
    for img in glob.glob(str(val_img / "*.jpg")):
        name = os.path.splitext(os.path.basename(img))[0]
        if name not in gt:
            gt[name] = False

    log(f"[eval] val images: {len(gt)} (fire={sum(gt.values())}, no_fire={len(gt)-sum(gt.values())})")

    model = YOLO(model_path)
    log(f"[eval] predicting val ({len(gt)} imgs) with workers=0 ...")

    try:
        results = model.predict(
            source=str(val_img), imgsz=640, conf=0.01, iou=0.45,
            device=DEVICE, workers=0, batch=8, stream=True,
            verbose=False, half=True,
        )
    except Exception as e:
        log(f"[eval] half=True failed ({e}), retry with half=False ...")
        results = model.predict(
            source=str(val_img), imgsz=640, conf=0.01, iou=0.45,
            device=DEVICE, workers=0, batch=8, stream=True,
            verbose=False,
        )

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

    # 扫描阈值
    best = None
    table = []
    for t in confs:
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
        table.append((t, P, R, F, tp, fp, fn))
        if best is None or F > best[1]:
            best = (t, F, P, R, tp, fp, fn)

    log("[eval] conf  P       R       F1      TP  FP  FN")
    for (t, P, R, F, tp, fp, fn) in table:
        log(f"       {t:.2f}  {P:.4f}  {R:.4f}  {F:.4f}  {tp:3d} {fp:3d} {fn:3d}")
    best_conf, best_F, best_P, best_R, _, _, _ = best
    log(f"[eval] BEST conf={best_conf:.2f} P={best_P:.4f} R={best_R:.4f} F1={best_F:.4f}")
    return best_conf, {"table": table, "best_conf": best_conf,
                       "P": best_P, "R": best_R, "F1": best_F}


def make_result_json(model_path, best_conf, out_json):
    """对验证集生成提交格式: {image_name(无扩展): 0/1}。"""
    val_img = OUT / "images" / "val"
    model = YOLO(model_path)
    try:
        results = model.predict(
            source=str(val_img), imgsz=640, conf=best_conf, iou=0.45,
            device=DEVICE, workers=0, batch=8, stream=True,
            verbose=False, half=True,
        )
    except Exception:
        results = model.predict(
            source=str(val_img), imgsz=640, conf=best_conf, iou=0.45,
            device=DEVICE, workers=0, batch=8, stream=True, verbose=False,
        )
    out = {}
    for r in results:
        name = os.path.splitext(os.path.basename(r.path))[0]
        boxes = r.boxes
        fire = False
        if boxes is not None and len(boxes) > 0:
            cls = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()
            fire = any(c == 0 and cf >= best_conf for c, cf in zip(cls, conf))
        out[name] = 1 if fire else 0
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[result.json] wrote {len(out)} entries -> {out_json}")
    return out


def main():
    log("=" * 60)
    log("[eval_only] 开始图像级火焰 P/R 评估 (跳过训练)")

    # 1. 找已有 best.pt
    best_pt, run_dir, n_epochs = find_best_pt()
    if best_pt is None:
        log("[eval_only] 没找到任何 best.pt, 无法评估")
        return

    # 2. 复制到统一位置
    dest = PIPE / "final_yolo26n_best.pt"
    shutil.copy(best_pt, dest)
    log(f"[eval_only] copied {best_pt} -> {dest}")

    # 3. 图像级阈值扫描
    conf, metrics = image_level_eval(str(dest), CONF_SWEEP)

    # 4. 生成验证集 result.json
    result_json = PIPE / "result_val_yolo26n.json"
    make_result_json(str(dest), conf, result_json)

    # 5. 写 RESULT.md
    P, R, F1 = metrics["P"], metrics["R"], metrics["F1"]
    lines = [
        "# 蒸馏训练结果汇总",
        "",
        f"- 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 教师: YOLOv11s (leeyunjai/yolo11-firedetect, Fire=0/Smoke=1)",
        f"- 蒸馏方式: 离线伪标签合并 + YOLOv26 标准训练(跨架构蒸馏)",
        f"- 训练状态: 已训练 {n_epochs} epochs (因系统内存限制未跑完 250 epoch, 用 best.pt 评估)",
        "",
        "## 图像级火焰 P/R（验证集, 仅评火焰类0）",
        "",
        "| 模型 | 最佳conf | Precision | Recall | F1 | TP | FP | FN |",
        "|---|---|---|---|---|---|---|---|",
        f"| yolo26n | {conf:.2f} | {P:.4f} | {R:.4f} | {F1:.4f} | {metrics['table'][[t[0] for t in metrics['table']].index(conf)][4]} | {metrics['table'][[t[0] for t in metrics['table']].index(conf)][5]} | {metrics['table'][[t[0] for t in metrics['table']].index(conf)][6]} |",
        "",
        "### 阈值扫描完整结果",
        "",
        "| conf | Precision | Recall | F1 | TP | FP | FN |",
        "|---|---|---|---|---|---|---|",
    ]
    for (t, p, r, f, tp, fp, fn) in metrics["table"]:
        marker = " **" if t == conf else ""
        lines.append(f"| {t:.2f}{marker} | {p:.4f} | {r:.4f} | {f:.4f} | {tp} | {fp} | {fn} |")
    lines.extend([
        "",
        f"## 推荐提交模型: yolo26n",
        f"- 权重: {dest}",
        f"- 训练 epochs: {n_epochs} (patience=40, 可能未完全收敛)",
        f"- 图像级火焰决策阈值 conf={conf:.2f}（类0框 conf>=此值 -> 图=1, 否则=0）",
        f"- 验证集 P={P:.4f} R={R:.4f} F1={F1:.4f}",
        f"- 验证集 result.json 示范: {result_json}",
        "",
        "## 决策逻辑",
        "1. 对每张图, 用模型推理得到检测框",
        "2. 仅取火焰框(类0), 若存在 conf>=阈值的火焰框 -> 图=1(有火)",
        "3. 烟雾框(类1)不触发'有火', 保护 Precision",
        "4. 无火焰框或全部低于阈值 -> 图=0(无火)",
        "",
        "## 下一步（竞赛提交）",
        "- 用 `infer_image_level.py` 对官方测试集推理, 输出 result.json (image_name -> 0/1)",
        "- 命令: `python infer_image_level.py --model final_yolo26n_best.pt --source <test_images> --conf {:.2f} --out result.json`".format(conf),
        "- 测试集提前3天公布, 下载后放入指定目录再推理即可",
        "- 如需进一步提升精度: 可在内存释放后继续训练(resume), 或用更大模型(yolo26s/m)",
    ])
    report = PIPE / "RESULT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    log(f"[eval_only] REPORT written: {report}")

    # 6. 更新 status.json
    status = {
        "stage": "DONE",
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "recommended": "yolo26n",
        "weights": str(dest),
        "best_conf": conf,
        "P": P, "R": R, "F1": F1,
        "epochs_trained": n_epochs,
        "note": "训练因系统内存限制未跑完250epoch, 用best.pt评估"
    }
    (PIPE / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    log("[eval_only] ALL DONE")
    log(f"[eval_only] 推荐: yolo26n, conf={conf:.2f}, P={P:.4f}, R={R:.4f}, F1={F1:.4f}")


if __name__ == "__main__":
    main()
