# -*- coding: utf-8 -*-
"""
纯D-Fire训练 yolo26n —— 官方1100一张不进训练。训练完自动在官方1100上做无泄漏评估。
铁律: 唯一时间戳目录(exist_ok=False) / workers=0 / cache流式 / CUDA-OOM自愈(empty_cache+降batch) / 心跳 / plots=False
输出: final_ext_yolo26n_best.pt + RESULT_ext.md + status_ext.json
断点续训: 设置环境变量 YD_RESUME=1 时, 会自动找最新的 ext_yolo26n_* 运行目录并从 last.pt 续训。
"""
import os, sys, time, shutil, datetime, subprocess, json
from pathlib import Path

for _k, _v in {
    "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4", "VECLIB_MAXIMUM_THREADS": "4",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}.items():
    os.environ.setdefault(_k, _v)

import torch
from ultralytics import YOLO

ROOT = Path(r"E:/火情监测/火情监测")
MODEL = ROOT / "yolo26n.pt"                        # COCO 预训练, 干净起点
DATA = ROOT / "distill_pipeline" / "external_train_dataset" / "dataset.yaml"
OUT_DIR = ROOT / "distill_pipeline" / "runs"
os.makedirs(OUT_DIR, exist_ok=True)
TS = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
NAME = "ext_yolo26n_" + TS
HB = ROOT / "distill_pipeline" / "ext_train.log"
FINAL = ROOT / "distill_pipeline" / "final_ext_yolo26n_best.pt"
EVAL = ROOT / "distill_pipeline" / "eval_official_clean.py"


def get_avail_phys_gb():
    """读当前可用物理内存(GB)。"""
    import ctypes
    class M(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = M(); m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 1e9, m.ullAvailPageFile / 1e9


def pick_batch_by_mem():
    """用户要求: 只用当前空余物理内存的一半。每batch约占0.06GB物理内存(经验值)。"""
    avail, pagefile = get_avail_phys_gb()
    target = avail * 0.5                          # 只用一半
    batch = max(8, min(24, int(target / 0.06)))   # 下限8保速度, 上限24防显存
    if batch % 2 == 1:
        batch += 1
    print("[mem] avail_phys=%.2fGB pagefile=%.2fGB target=%.2fGB -> batch=%d" %
          (avail, pagefile, target, batch), flush=True)
    return batch


def find_latest_run():
    """找最新的 ext_yolo26n_* 训练目录(含 last.pt), 用于断点续训。"""
    runs = sorted([d for d in OUT_DIR.glob("ext_yolo26n_*") if d.is_dir()],
                  key=lambda p: p.name, reverse=True)
    for d in runs:
        last = d / "weights" / "last.pt"
        if last.exists():
            return d, last
    return None, None


def heartbeat(trainer):
    ep = trainer.epoch + 1
    if ep % 5 == 0:
        avail, pf = get_avail_phys_gb()
        with open(HB, "a", encoding="utf-8") as f:
            f.write("%s epoch=%d  avail_phys=%.2fGB\n" % (time.strftime("%H:%M:%S"), ep, avail))


def train_once(batch, resume_run=None, resume_last=None):
    if resume_last is not None:
        # 断点续训: 沿用上次运行的超参与权重
        model = YOLO(str(resume_last))
        model.add_callback("on_train_epoch_end", heartbeat)
        model.train(resume=True)
        best = resume_run / "weights" / "best.pt"
    else:
        model = YOLO(str(MODEL))
        model.add_callback("on_train_epoch_end", heartbeat)
        model.train(
            data=str(DATA),
            epochs=40, patience=15, imgsz=640,
            batch=batch, workers=0, cache=False,
            amp=True, cos_lr=True, close_mosaic=10,
            device=0, lr0=0.002, lrf=0.01,
            name=NAME, project=str(OUT_DIR), exist_ok=False,
            plots=False,
        )
        best = OUT_DIR / NAME / "weights" / "best.pt"
    if not best.exists():
        cand = list(best.parent.glob("*.pt"))
        best = max(cand, key=lambda p: p.stat().st_mtime) if cand else None
    return best


def parse_best_line(out):
    info = {}
    for line in out.splitlines():
        if line.startswith(">>> BEST"):
            for p in line.replace(">>> BEST", "").split():
                if "=" in p:
                    k, v = p.split("=", 1)
                    info[k] = v
    return info


def main():
    open(HB, "w", encoding="utf-8").write("start %s\n" % time.strftime("%H:%M:%S"))
    mem_batch = pick_batch_by_mem()

    # 断点续训判定
    resume_run, resume_last = (None, None)
    if os.environ.get("YD_RESUME") == "1":
        resume_run, resume_last = find_latest_run()
        if resume_last:
            print("[main] RESUME mode from %s" % resume_last, flush=True)

    batch_env = int(os.environ.get("YD_BATCH_N", str(mem_batch)))
    start_batch = min(batch_env, mem_batch)        # 取二者较小, 不超内存
    batches = [start_batch, max(8, start_batch - 8), 8, 4]
    seen = set(); batches = [b for b in batches if not (b in seen or seen.add(b))]
    print("[main] start_batch=%d 降级链=%s" % (start_batch, batches), flush=True)

    best = None
    if resume_last is not None:
        # 续训只试一次(沿用保存的batch); 失败则回退到全新训练
        try:
            print("[train] resume batch=%d" % start_batch, flush=True)
            best = train_once(start_batch, resume_run, resume_last)
            print("[train] resume done best=%s" % best, flush=True)
        except Exception as e:
            print("[train] resume FAILED %r, fallback to fresh" % e, flush=True)
            best = None
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            time.sleep(10)

    if best is None or not (best.exists() if best else False):
        for b in batches:
            try:
                print("[train] try batch=%d" % b, flush=True)
                best = train_once(b)
                print("[train] done batch=%d best=%s" % (b, best), flush=True)
                break
            except Exception as e:
                avail, pf = get_avail_phys_gb()
                print("[train] Exception %r at batch=%d (avail=%.2fGB) cleanup+descend" %
                      (e, b, avail), flush=True)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                time.sleep(10)
                continue

    if best is None or not best.exists():
        print("[train] FATAL: no weights produced", flush=True)
        sys.exit(1)
    shutil.copy(best, FINAL)
    print("[train] copied -> %s" % FINAL, flush=True)

    # 自动在官方1100上做无泄漏评估
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    print("[eval] start on official 1100 (no-leak)", flush=True)
    try:
        r = subprocess.run([sys.executable, str(EVAL), str(FINAL)],
                           capture_output=True, text=True, timeout=1800)
        out = (r.stdout or "") + "\n" + (r.stderr or "")
    except Exception as e:
        out = "[eval] FAILED: %r" % e
    print(out, flush=True)

    # 写报告
    with open(ROOT / "distill_pipeline" / "RESULT_ext.md", "w", encoding="utf-8") as f:
        f.write("# 纯D-Fire训练 yolo26n — 官方1100无泄漏评估\n\n")
        f.write("- 训练集: `external_train_dataset` (纯D-Fire 17,525张, **官方1100零进入**)\n")
        f.write("- 起点: COCO 预训练 yolo26n.pt (干净, 未见过官方图)\n")
        f.write("- 评估: `eval_official_clean.py` 在官方完整1100张上, 纯官方标注当GT, **无泄漏**\n\n")
        f.write("```\n" + out + "\n```\n")

    info = {"stage": "DONE", "model": "final_ext_yolo26n_best.pt",
            "note": "官方1100无泄漏; 训练仅用纯D-Fire"}
    try:
        d = parse_best_line(out)
        info["best_conf"] = float(d.get("conf", 0.1))
        info["P"] = float(d.get("P", 0))
        info["R"] = float(d.get("R", 0))
        info["F1"] = float(d.get("F1", 0))
        info["TP"] = int(d.get("TP", 0))
        info["FP"] = int(d.get("FP", 0))
        info["FN"] = int(d.get("FN", 0))
    except Exception:
        pass
    with open(ROOT / "distill_pipeline" / "status_ext.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print("[train] DONE", flush=True)


if __name__ == "__main__":
    main()
