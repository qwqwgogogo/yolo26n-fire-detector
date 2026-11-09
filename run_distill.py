# -*- coding: utf-8 -*-
"""
run_distill.py —— 离线知识蒸馏第 2 步：训练 YOLOv26 学生并做图像级火焰 P/R 阈值扫描。

特性:
- 学生尺寸: n(主, 部署友好) 与 s(备选, 精度更高)，均从 COCO 预训练权重起步。
- 自愈合训练: OOM 自动降 imgsz -> 降 workers 重试；其它异常降 workers 重试；最多多轮。
- 跑满资源: batch=-1(自动占满显存) + workers=20(喂饱 CPU) + device=0 + amp。
- 图像级评估: 仅评"火焰(类0)"，输出每张图 0/1；扫描 conf 阈值挑 F1 最优工作点。
- 产出: 每个尺寸的最佳权重副本 + 阈值 + 验证集 result.json 示范 + RESULT.md 汇总。

状态文件 distill_pipeline/status.json: {"stage": "RUNNING|DONE|FAILED", ...}
"""
import os, sys, json, glob, shutil, time, math
from pathlib import Path

ROOT = Path(r"E:/火情监测/火情监测")
PIPE = ROOT / "distill_pipeline"
OUT = PIPE / "distill_dataset"
RUNS = PIPE / "runs"
STATUS = PIPE / "status.json"
REPORT = PIPE / "RESULT.md"
LOG = PIPE / "run_distill.log"

# 限制线程数: 笔记本物理内存/页面文件较小, 32 核下 OpenBLAS 默认开满线程会提交过量内存
# 导致 import torch 失败(WinError 1114/1455)。必须在 import torch 前设置。
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
WORKERS = 0  # 必须为0: Windows 页面文件太小, 任何>0 的 dataloader 子进程都会重新 import torch 加载 CUDA DLL 而崩(WinError 1455)
IMGSZ_CHAIN = [640, 512, 416]
EPOCHS = 250
PATIENCE = 40
CLOSE_MOSAIC = 20
CONF_SWEEP = [round(x * 0.05, 2) for x in range(2, 19)]  # 0.10 .. 0.90


def log(s):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + str(s)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def set_status(stage, **kw):
    d = {"stage": stage, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    d.update(kw)
    STATUS.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")


def train_size(size, resume_dir=None):
    """自愈合训练单个尺寸, 返回最佳权重路径或 None。

    关键设计:
    1. 续训: resume_dir 提供时, 从该目录 last.pt 继续(省去从头训的耗时, 复用已训 epoch)。
    2. 内存自适应降级链 BATCH_CHAIN: 遇 MemoryError(系统RAM耗尽, 上次 epoch160 崩的真凶)
       自动降 batch 重试; 起始 batch 通过 YD_BATCH_N/YD_BATCH_S 环境变量可被监督器动态调低。
       起始 batch 取较高值以利用显存提升 GPU 利用率(workers=0 时唯一安全杠杆)。
    3. 从头训: 唯一时间戳目录名, 避免 safe-delete 拦截。
    4. 心跳回调: 每 5 epoch 写日志, 防监督器误判卡死。
    """
    best_weights = None
    # 起始 batch: 环境变量可覆盖(监督器据内存动态调低); n 默认 32 利用显存提 GPU 利用率, s 默认 8
    if size == "n":
        BATCH_CHAIN = [int(os.environ.get("YD_BATCH_N", "32")), 24, 16, 8, 4]
    else:
        BATCH_CHAIN = [int(os.environ.get("YD_BATCH_S", "8")), 6, 4, 2]
    WORKERS = 0  # 必须为0: 任何>0 dataloader 子进程重加载 CUDA DLL 必崩(WinError 1455)

    def _heartbeat(trainer):
        ep = trainer.epoch + 1
        if ep % 5 == 0 or ep == 1:
            log(f"[heartbeat] yolo26{size} epoch {ep}/{trainer.epochs}")

    # ---- 续训模式 ----
    if resume_dir is not None:
        last = Path(resume_dir) / "weights" / "last.pt"
        if last.exists():
            for bi, batch in enumerate(BATCH_CHAIN):
                try:
                    log(f"[train yolo26{size}] 续训 resume={resume_dir} batch={batch}")
                    model = YOLO(str(last))
                    model.add_callback("on_train_epoch_end", _heartbeat)
                    model.train(resume=True, batch=batch)
                    cand = Path(resume_dir) / "weights" / "best.pt"
                    if cand.exists():
                        best_weights = str(cand)
                        log(f"[train yolo26{size}] resume OK best={best_weights}")
                        return best_weights
                    if last.exists():
                        best_weights = str(last)
                        log(f"[train yolo26{size}] resume OK last={best_weights}")
                        return best_weights
                except MemoryError as e:
                    log(f"[train yolo26{size}] resume MemoryError batch={batch} -> 降 batch 从头训")
                    break  # 跳到从头训, 从该 batch 继续降级
                except Exception as e:
                    log(f"[train yolo26{size}] resume 失败: {repr(e)} -> 改从头训")
                    break

    # ---- 从头训练(唯一时间戳目录名, batch 降级链) ----
    for bi, batch in enumerate(BATCH_CHAIN):
        for zi, imgsz in enumerate(IMGSZ_CHAIN):
            workers = WORKERS
            ok = False
            last_name = None
            for attempt in range(6):
                run_name = f"distill_yolo26{size}_{int(time.time() * 1000)}_{zi}_{attempt}"
                last_name = run_name
                try:
                    log(f"[train yolo26{size}] imgsz={imgsz} batch={batch} workers={workers} attempt={attempt} name={run_name}")
                    model = YOLO(str(ROOT / f"yolo26{size}.pt"))
                    model.add_callback("on_train_epoch_end", _heartbeat)
                    model.train(
                        data=str(OUT / "dataset.yaml"),
                        epochs=EPOCHS, patience=PATIENCE,
                        imgsz=imgsz, batch=batch, workers=workers,
                        device=DEVICE, amp=True, cos_lr=True,
                        close_mosaic=CLOSE_MOSAIC,
                        name=run_name, project=str(RUNS),
                        exist_ok=True, verbose=False,
                    )
                    ok = True
                    break
                except MemoryError as e:
                    log(f"[train yolo26{size}] MemoryError batch={batch} imgsz={imgsz} -> 降 batch 重试")
                    break  # 跳到下一个更小 batch
                except RuntimeError as e:
                    s = str(e).lower()
                    if "out of memory" in s:
                        if zi < len(IMGSZ_CHAIN) - 1:
                            log(f"[train yolo26{size}] VRAM OOM imgsz={imgsz} -> 尝试更小 imgsz")
                            break
                        log(f"[train yolo26{size}] VRAM OOM at min imgsz, 放弃该尺寸")
                        break
                    log(f"[train yolo26{size}] RuntimeError: {e}")
                    if workers > 0:
                        workers = max(0, workers // 2)
                        continue
                    break
                except Exception as e:
                    es = str(e)
                    if ("SAFE_DELETE" in es) or ("safe-delete" in es):
                        log(f"[train yolo26{size}] safe-delete 拦截, 用新目录名重试 (attempt {attempt})")
                        continue
                    if ("页面文件" in es) or ("1455" in es) or ("out of memory" in es.lower()):
                        if workers > 0:
                            log(f"[train yolo26{size}] 页面文件/OS 错误 -> workers 降为 0")
                            workers = 0
                            continue
                        log(f"[train yolo26{size}] 页面文件错误且 workers=0, 无法继续")
                        break
                    log(f"[train yolo26{size}] Exception: {repr(e)}")
                    if workers > 0:
                        workers = max(0, workers // 2)
                        continue
                    break
            if ok:
                cand = RUNS / last_name / "weights" / "best.pt"
                if cand.exists():
                    best_weights = str(cand)
                    log(f"[train yolo26{size}] OK best={best_weights}")
                    return best_weights
                else:
                    log(f"[train yolo26{size}] finished but best.pt missing?!")
                    last = RUNS / last_name / "weights" / "last.pt"
                    if last.exists():
                        best_weights = str(last)
                        return best_weights
    return best_weights


def find_best_partial(size):
    """找 yolo26{size} 已存在的部分训练目录中 epoch 数最多的那个(用于续训)。"""
    import glob as _glob
    best = None
    best_ep = -1
    for r in _glob.glob(str(RUNS / f"distill_yolo26{size}_*")):
        rc = Path(r) / "results.csv"
        if rc.exists():
            try:
                with open(rc, encoding="utf-8") as f:
                    n = sum(1 for _ in f) - 1
            except Exception:
                n = 0
            if n > best_ep:
                best_ep = n
                best = r
    return best


def image_level_eval(model_path, confs):
    """用最佳权重在验证集上做图像级火焰 P/R 扫描。返回 (best_conf, metrics_dict)。"""
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

    model = YOLO(model_path)
    log(f"[eval] predicting val ({len(gt)} imgs) ...")
    try:
        results = model.predict(
            source=str(val_img), imgsz=640, conf=0.01, iou=0.45,
            device=DEVICE, workers=0, batch=16, stream=True,
            verbose=False, half=True,
        )
    except Exception:
        results = model.predict(
            source=str(val_img), imgsz=640, conf=0.01, iou=0.45,
            device=DEVICE, workers=0, batch=16, stream=True,
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
            best = (t, F, P, R)
    log("[eval] conf  P       R       F1")
    for (t, P, R, F, *_ ) in table:
        log(f"       {t:.2f}  {P:.4f}  {R:.4f}  {F:.4f}")
    best_conf, best_F, best_P, best_R = best
    log(f"[eval] BEST conf={best_conf:.2f} P={best_P:.4f} R={best_R:.4f} F1={best_F:.4f}")
    return best_conf, {"table": table, "best_conf": best_conf,
                       "P": best_P, "R": best_R, "F1": best_F}


def make_result_json(model_path, best_conf, out_json):
    """对验证集生成提交格式示范: {image_name(无扩展): 0/1}。"""
    val_img = OUT / "images" / "val"
    model = YOLO(model_path)
    try:
        results = model.predict(
            source=str(val_img), imgsz=640, conf=best_conf, iou=0.45,
            device=DEVICE, workers=0, batch=16, stream=True,
            verbose=False, half=True,
        )
    except Exception:
        results = model.predict(
            source=str(val_img), imgsz=640, conf=best_conf, iou=0.45,
            device=DEVICE, workers=0, batch=16, stream=True, verbose=False,
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


def main():
    set_status("RUNNING", step="gen_pseudo")
    sys.path.insert(0, str(PIPE))
    import gen_pseudo
    if (OUT / ".pseudo_done").exists():
        log("[main] 伪标签已就绪(.pseudo_done), 跳过 gen_pseudo")
    else:
        gen_pseudo.main()

    set_status("RUNNING", step="train")
    sizes = ["n"]  # 仅 yolo26n: 用户要求一小时内完成 26N; yolo26s 暂不跑(耗时2.5h+)
    results = {}
    final_best = None
    for size in sizes:
        # yolo26n 续训已有 110-epoch 模型(省时); yolo26s 从头训
        rdir = find_best_partial(size) if size == "n" else None
        if rdir:
            log(f"[main] yolo26{size} 发现部分训练 {rdir}, 续训")
        w = train_size(size, resume_dir=rdir)
        if w is None:
            log(f"[main] yolo26{size} training failed, skip")
            continue
        # 复制最佳权重到统一位置
        dest = PIPE / f"final_yolo26{size}_best.pt"
        shutil.copy(w, dest)
        log(f"[main] copied best -> {dest}")
        conf, metrics = image_level_eval(w, CONF_SWEEP)
        results[size] = {
            "weights": str(dest), "best_conf": conf,
            "P": metrics["P"], "R": metrics["R"], "F1": metrics["F1"],
        }
        # 验证集 result.json 示范
        make_result_json(str(dest), conf, PIPE / f"result_val_yolo26{size}.json")
        # 选全局最优(F1)
        if final_best is None or metrics["F1"] > results[final_best]["F1"]:
            final_best = size

    if not results:
        set_status("FAILED", reason="all trainings failed")
        log("[main] ALL FAILED")
        return

    # 汇总报告
    fb = results[final_best]
    lines = ["# 蒸馏训练结果汇总", ""]
    lines.append(f"- 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 教师: YOLOv11 (firedetect-11x/11s)  Fire=0, Smoke=1")
    lines.append(f"- 蒸馏方式: 离线伪标签合并 + YOLOv26 标准训练(跨架构蒸馏)")
    lines.append("")
    lines.append("## 各尺寸图像级火焰 P/R（验证集，仅评火焰类0）")
    lines.append("")
    lines.append("| 尺寸 | 最佳conf | Precision | Recall | F1 | 权重 |")
    lines.append("|---|---|---|---|---|---|")
    for sz, r in results.items():
        lines.append(f"| yolo26{sz} | {r['best_conf']:.2f} | {r['P']:.4f} | {r['R']:.4f} | {r['F1']:.4f} | {r['weights']} |")
    lines.append("")
    lines.append(f"## 推荐提交模型: yolo26{final_best}")
    lines.append(f"- 权重: {fb['weights']}")
    lines.append(f"- 图像级火焰决策阈值 conf={fb['best_conf']:.2f}（类0框 conf>=此值 -> 图=1）")
    lines.append(f"- 验证集 P={fb['P']:.4f} R={fb['R']:.4f} F1={fb['F1']:.4f}")
    lines.append("")
    lines.append("## 下一步（竞赛提交）")
    lines.append("- 用 `infer_image_level.py` 对官方测试集推理，输出 result.json（image_name -> 0/1）。")
    lines.append("- 仅火焰(类0)触发'有火'；烟雾(类1)不得触发，以免影响 Precision。")
    lines.append("- 测试集提前3天公布，下载后放入指定目录再推理即可。")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log("[main] REPORT written: " + str(REPORT))

    set_status("DONE", recommended=f"yolo26{final_best}",
               weights=str(fb["weights"]), best_conf=fb["best_conf"],
               P=fb["P"], R=fb["R"], F1=fb["F1"])
    log("[main] ALL DONE")


if __name__ == "__main__":
    main()
