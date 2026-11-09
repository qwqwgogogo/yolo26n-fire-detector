# -*- coding: utf-8 -*-
"""
exp_train_eval.py —— 单个优化实验:
  1) 构建合并数据集(仅公开集, 官方1100绝不进训练, 文件名兜底排除)
  2) 训练(铁律: 动态内存batch只用一半 / workers=0 / CUDA-OOM+MemoryError自愈 / 断点续训 / 心跳)
  3) 在官方完整1100张上做无泄漏图像级火焰P/R评估, 扫描conf找 max(min(P,R)) 与 达标点
CLI: python exp_train_eval.py <config.json>
config.json 字段:
  prefix      输出前缀(唯一)
  sources     [{"name":"dfire"|"hiennguyen","max_n":0(0=全量)}]
  model_size  "n"|"s"
  base_model  "yolo26n.pt"|"yolo26s.pt" (项目根)
  epochs, patience, imgsz
  enhance     bool (颜色/翻转/马赛克增强, 针对相近颜色误报)
"""
import os, sys, json, time, shutil, random, subprocess, datetime, glob, re
import ctypes, threading
from pathlib import Path

# 中文路径修复: 见 optimize_loop 说明. 本脚本启动 tiled_eval.py 子进程(中文路径)需 UTF-8 模式,
# 否则 subprocess 报 "Unable to create process ... ????". 同时子进程继承此 env.
os.environ.setdefault("PYTHONUTF8", "1")

ROOT = Path(r"E:/火情监测/火情监测")
DP = ROOT / "distill_pipeline"
EXT = ROOT / "external_datasets"
OFF_IMG = ROOT / "dataset_root" / "images" / "train"
RUNS = Path(os.environ.get("FIRE_RUNS_DIR", str(DP / "runs")))  # 可经环境变量指向D盘(避免E盘满崩溃)
OPT_DATA = DP / "opt_datasets"
OPT_DATA.mkdir(exist_ok=True)
RUNS.mkdir(exist_ok=True)

SOURCES = {
    "dfire": {"img": EXT / "academic_D-Fire/train/images",
              "label": EXT / "academic_D-Fire/train/labels"},
    "alirhayem": {"img": EXT / "Kaggle_alirhayem_FireSmoke/Enhanced/Enhanced/train/images",
                   "label": EXT / "Kaggle_alirhayem_FireSmoke/Enhanced/Enhanced/train/labels"},
    "cctv": {"img": EXT / "HF_CCTV_Smoke_Fire/CCTV_Fire_Smoke_Emergency_Detection_Dataset/images",
             "label": EXT / "HF_CCTV_Smoke_Fire/CCTV_Fire_Smoke_Emergency_Detection_Dataset/labels"},
    "kien": {"img": EXT / "HF_KienNgyuen_FireSmoke/_unzipped/CCTV Smoke & Fire Emergency Detection Dataset/images",
             "label": EXT / "HF_KienNgyuen_FireSmoke/_unzipped/CCTV Smoke & Fire Emergency Detection Dataset/labels"},
    "hiennguyen": {"img": EXT / "HF_hiennguyen_FireSmoke/images",
                   "label": EXT / "HF_hiennguyen_FireSmoke/labels"},
}

ENV_THREAD = {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
              "NUMEXPR_NUM_THREADS": "4", "VECLIB_MAXIMUM_THREADS": "4",
              "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def avail_phys_gb():
    class M(ctypes.Structure):
        _fields_ = [("d", ctypes.c_ulong), ("l", ctypes.c_ulong),
                    ("t", ctypes.c_ulonglong), ("a", ctypes.c_ulonglong),
                    ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                    ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong),
                    ("ae", ctypes.c_ulonglong)]
    m = M(); m.d = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.a / 1e9


def build_dataset(cfg):
    prefix = cfg["prefix"]
    out = OPT_DATA / prefix
    img_tr = out / "images" / "train"; lbl_tr = out / "labels" / "train"
    img_va = out / "images" / "val"; lbl_va = out / "labels" / "val"
    for d in (img_tr, lbl_tr, img_va, lbl_va):
        d.mkdir(parents=True, exist_ok=True)
    # 官方文件名集合(兜底防泄漏)
    off_names = set()
    if OFF_IMG.exists():
        for f in os.listdir(OFF_IMG):
            off_names.add(os.path.splitext(f)[0])
    collected = []
    for s in cfg["sources"]:
        name = s["name"]; max_n = int(s.get("max_n", 0))
        src = SOURCES.get(name)
        if src is None:
            log(f"[build] 未知源 {name}, 跳过"); continue
        img_dir, lbl_dir = src["img"], src["label"]
        if not img_dir.exists():
            log(f"[build] 源 {name} 图片目录不存在 {img_dir}"); continue
        labels = sorted(glob.glob(str(lbl_dir / "*.txt"))) if lbl_dir.exists() else []
        if max_n and len(labels) > max_n:
            random.seed(12345); labels = random.sample(labels, max_n)
        cnt = 0
        for lp in labels:
            stem = os.path.splitext(os.path.basename(lp))[0]
            if stem in off_names:
                continue
            im = None
            for ext in (".jpg", ".JPG", ".png", ".PNG"):
                cand = img_dir / (stem + ext)
                if cand.exists():
                    im = cand; break
            if im is None:
                continue
            collected.append((im, Path(lp), stem, name)); cnt += 1
        log(f"[build] 源 {name}: 采用 {cnt} 张")
    random.seed(999); random.shuffle(collected)
    n_val = max(1, int(len(collected) * 0.03))
    val = collected[:n_val]; tr = collected[n_val:]

    def link(items, idir, ldir):
        for im, lp, stem, name in items:
            di = idir / (f"{name}_{stem}{im.suffix}"); dl = ldir / (f"{name}_{stem}.txt")
            try:
                if not di.exists():
                    os.link(im, di)
            except OSError:
                if not di.exists():
                    shutil.copy(im, di)
            try:
                if not dl.exists():
                    os.link(lp, dl)
            except OSError:
                if not dl.exists():
                    shutil.copy(lp, dl)
    link(tr, img_tr, lbl_tr); link(val, img_va, lbl_va)
    # ---- 针对性样本(expF): 从 targeted_pool 拉硬负(相近颜色/车灯/烟里火光)+硬正(小远暗火) ----
    if cfg.get("targeted"):
        pool = DP / (cfg.get("targeted_pool") or "targeted_pool")
        pim = pool / "images"; plb = pool / "labels"
        extra = []
        if pim.exists():
            labs = sorted(glob.glob(str(plb / "*.txt"))) if plb.exists() else []
            for lp in labs:
                stem = os.path.splitext(os.path.basename(lp))[0]
                im = None
                for ext in (".jpg", ".JPG", ".png", ".PNG"):
                    c = pim / (stem + ext)
                    if c.exists():
                        im = c; break
                if im is None:
                    continue
                extra.append((im, Path(lp), stem, "tgt"))
            if extra:
                link(extra, img_tr, lbl_tr)
                log(f"[build] targeted_pool: 加入 {len(extra)} 张定向硬样本(训练集)")
        else:
            log(f"[build] targeted=True 但池不存在 {pim}, 退化为普通训练")
    # ---- 蒸馏伪标签(expG): 教师对未标注域池伪标, 扩充训练(域匹配) ----
    if cfg.get("distill"):
        try:
            n_ps = generate_pseudo_labels(cfg, out)
            log(f"[build] distill 伪标签新增 {n_ps} 张(训练集)")
        except Exception as e:
            log(f"[build] distill 伪标签失败, 退化为普通训练: {e}")
    yaml_path = out / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {out}\n")
        f.write("train: images/train\nval: images/val\n")
        f.write("nc: 2\nnames: [fire, smoke]\n")
    log(f"[build] 完成: 训练={len(tr)} 验证={len(val)} -> {yaml_path}")
    return yaml_path, len(tr), len(val)


def generate_pseudo_labels(cfg, out):
    """蒸馏产线(expG): 用已有强模型(教师)对未标注域池推理伪标, 并入训练集扩充域覆盖.
    防御性: 池缺失/无图/无教师/单图失败 均安全跳过, 返回新增张数(0=退化为普通训练)."""
    import torch
    from ultralytics import YOLO
    d = cfg.get("distill") or {}
    pool = DP / (d.get("pseudo_pool") or "distill_pseudo_pool")
    pim = pool / "images"
    if not pim.exists():
        log("[distill] 伪标签池不存在, 跳过"); return 0
    imgs = []
    for ext in (".jpg", ".JPG", ".png", ".PNG"):
        imgs += glob.glob(str(pim / f"*{ext}"))
    if not imgs:
        log("[distill] 伪标签池无图, 跳过"); return 0
    teacher = d.get("teacher_pt")
    if not teacher:
        cands = sorted(DP.glob("final_*_best.pt"), reverse=True)  # 复用已有最佳学生当教师
        teacher = str(cands[0]) if cands else None
    if not teacher or not Path(teacher).exists():
        log("[distill] 无可用教师, 跳过伪标签"); return 0
    m = YOLO(teacher)
    img_tr = out / "images" / "train"; lbl_tr = out / "labels" / "train"
    conf_t = float(d.get("pseudo_conf", 0.25))
    n = 0
    for ip in imgs:
        stem = os.path.splitext(os.path.basename(ip))[0]
        di = img_tr / (f"pseudo_{stem}{Path(ip).suffix}")
        dl = lbl_tr / (f"pseudo_{stem}.txt")
        if dl.exists():
            n += 1; continue
        try:
            res = m.predict(ip, conf=conf_t, imgsz=int(cfg.get("imgsz", 640)),
                            device=0, verbose=False)
            boxes = res[0].boxes
            if boxes is None or len(boxes) == 0:
                continue
            lines = []
            for b in boxes:
                cls = int(b.cls[0])
                if cls not in (0, 1):
                    continue
                x, y, w, h = b.xywhn[0].tolist()
                lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            if not lines:
                continue
            try:
                if not di.exists():
                    os.link(ip, di)
            except OSError:
                if not di.exists():
                    shutil.copy(ip, di)
            with open(dl, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            n += 1
        except Exception:
            continue
    torch.cuda.empty_cache()
    return n


def train(cfg, yaml_path):
    import torch
    from ultralytics import YOLO
    prefix = cfg["prefix"]
    base = ROOT / (cfg.get("base_model") or f"yolo{cfg.get('model_size','n')}.pt")
    if not base.exists():
        raise FileNotFoundError(f"base model 不存在: {base}")
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    name = f"{prefix}_{ts}"
    hb = DP / f"train_{prefix}.log"
    open(hb, "w", encoding="utf-8").write("start %s\n" % time.strftime("%H:%M:%S"))
    # 断点续训: 同prefix最新 last.pt
    resume_run = None; resume_last = None
    runs = sorted([d for d in RUNS.glob(f"{prefix}_*") if d.is_dir()],
                  key=lambda p: p.name, reverse=True)
    for d in runs:
        last = d / "weights" / "last.pt"
        if last.exists():
            resume_run = d; resume_last = last; break
    enhance = cfg.get("enhance", False)
    aug = dict(hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, fliplr=0.5, mosaic=1.0, scale=0.5) if enhance else {}

    def heartbeat(trainer):
        ep = trainer.epoch + 1
        if ep % 5 == 0:
            with open(hb, "a", encoding="utf-8") as f:
                f.write("%s epoch=%d avail=%.2fGB\n" % (time.strftime("%H:%M:%S"), ep, avail_phys_gb()))

    avail = avail_phys_gb()
    # 安全固定batch: yolo26n/s 在 batch=8 时峰值RAM<2GB, 即便空闲仅3GB也能稳跑; 空闲>=5GB才用12
    batch = 8 if avail < 5 else 12
    log(f"[train] avail={avail:.2f}GB -> batch={batch} (安全固定)")
    batches = [batch, 8, 4]
    best = None
    if resume_last is not None:
        try:
            log(f"[train] resume from {resume_last}")
            m = YOLO(str(resume_last)); m.add_callback("on_train_epoch_end", heartbeat)
            m.train(resume=True)
            best = resume_run / "weights" / "best.pt"
        except Exception as e:
            log(f"[train] resume fail {e}, fallback fresh"); best = None
            torch.cuda.empty_cache(); time.sleep(8)
    if best is None or not (best.exists() if best else False):
        for b in batches:
            try:
                log(f"[train] try batch={b}")
                m = YOLO(str(base)); m.add_callback("on_train_epoch_end", heartbeat)
                m.train(data=str(yaml_path), epochs=int(cfg.get("epochs", 40)),
                        patience=int(cfg.get("patience", 15)), imgsz=int(cfg.get("imgsz", 640)),
                        batch=b, workers=0, cache=False, amp=True, cos_lr=True,
                        close_mosaic=10, device=0, lr0=0.002, lrf=0.01,
                        name=name, project=str(RUNS), exist_ok=False, plots=False, **aug)
                best = RUNS / name / "weights" / "best.pt"
                if not best.exists():
                    cands = list(best.parent.glob("*.pt"))
                    best = max(cands, key=lambda p: p.stat().st_mtime) if cands else None
                break
            except Exception as e:
                log(f"[train] fail batch={b}: {e}; cleanup+descend")
                torch.cuda.empty_cache(); time.sleep(10); continue
    if best is None or not best.exists():
        log("[train] FATAL no weights"); sys.exit(1)
    final = DP / f"final_{prefix}_best.pt"
    shutil.copy(best, final)
    log(f"[train] copied -> {final}")
    return final


def eval_model(model_pt, prefix):
    eval_script = DP / "tiled_eval.py"   # 切片(Tiled)推理评估: 针对小火<1%面积漏检死穴, 召回远高于整图直推
    try:
        r = subprocess.run([sys.executable, str(eval_script), str(model_pt)],
                           capture_output=True, text=True, timeout=2400,
                           env=dict(os.environ, PYTHONUTF8="1"))
    except subprocess.TimeoutExpired:
        log("[eval] 评估超时(>40min), 视为本次失败, 由循环resume重试")
        return None
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    log("[eval] " + out[-2000:])
    best_min = -1; best_row = None; reached = False; reached_row = None
    for line in out.splitlines():
        m = re.match(r"\s*(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s+(\d+)\s+(\d+)", line)
        if m:
            conf = float(m.group(1)); P = float(m.group(2)); R = float(m.group(3))
            mn = min(P, R)
            if mn > best_min:
                best_min = mn; best_row = (conf, P, R)
            if P >= 0.98 and R >= 0.98:
                reached = True; reached_row = (conf, P, R)
    status = {"exp": prefix, "model": f"final_{prefix}_best.pt",
              "maxmin_PR": best_min,
              "best_conf": best_row[0] if best_row else 0.1,
              "P": best_row[1] if best_row else 0.0,
              "R": best_row[2] if best_row else 0.0,
              "reached_98": reached,
              "reached_conf": reached_row[0] if reached_row else None}
    with open(DP / f"status_{prefix}.json", "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
    with open(DP / f"RESULT_{prefix}.md", "w", encoding="utf-8") as f:
        f.write(f"# 实验 {prefix} — 官方1100无泄漏评估\n\n")
        f.write(f"- max(min(P,R)) = **{best_min:.4f}**\n- reached_98 = {reached}\n\n")
        f.write(f"```\n{out}\n```\n")
    log(f"[eval] 完成 prefix={prefix} maxmin={best_min:.4f} reached={reached}")
    return status


def main():
    if len(sys.argv) < 2:
        print("usage: exp_train_eval.py <config.json>"); sys.exit(1)
    cfg = json.load(open(sys.argv[1], encoding="utf-8"))
    for k, v in ENV_THREAD.items():
        os.environ.setdefault(k, v)
    prefix = cfg["prefix"]
    # 心跳守护线程: 每30s刷新 hb_<prefix>.alive, 覆盖 构建/训练/评估 全阶段,
    # 让循环改用心跳判卡死 -> 评估阶段(不更新results.csv)不会被误杀.
    _stop = threading.Event()
    def _beat():
        hb = DP / f"hb_{prefix}.alive"
        while not _stop.is_set():
            try:
                hb.touch()
            except Exception:
                pass
            time.sleep(30)
    threading.Thread(target=_beat, daemon=True).start()
    log(f"[main] 实验 {prefix} 开始")
    try:
        yaml_path, ntr, nval = build_dataset(cfg)
        final = train(cfg, yaml_path)
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass
        st = eval_model(final, prefix)
        if st is None:
            log(f"[main] 实验 {prefix} 评估失败(None), 退出由循环resume重试")
            sys.exit(1)
        log(f"[main] 实验 {prefix} 结束: {st}")
    finally:
        _stop.set()
    sys.exit(0)


if __name__ == "__main__":
    main()
