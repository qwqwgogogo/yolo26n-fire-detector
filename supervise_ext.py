# -*- coding: utf-8 -*-
"""
监督器: 确保「纯D-Fire训练 yolo26n → 官方1100无泄漏评估」跑到完成。
- 每 POLL 秒检查一次
- 仅当 last.pt 陈旧(>STALE) 且 GPU 无计算进程 时才重启, 杜绝重复启动
- 训练崩溃 -> 带 YD_RESUME=1 续训重启
- 权重已产出但 status 非 DONE(评估可能失败) -> 补跑评估
- status_ext.json 显示 DONE -> 退出
用法: python supervise_ext.py  (后台运行)
"""
import os, time, subprocess, sys, json, glob

ROOT = "E:/火情监测/火情监测"
STATUS = os.path.join(ROOT, "distill_pipeline/status_ext.json")
TRAIN = os.path.join(ROOT, "distill_pipeline/train_external.py")
EVAL = os.path.join(ROOT, "distill_pipeline/eval_official_clean.py")
FINAL = os.path.join(ROOT, "distill_pipeline/final_ext_yolo26n_best.pt")
LOG = os.path.join(ROOT, "distill_pipeline/train_ext.log")
RUNS = os.path.join(ROOT, "distill_pipeline/runs")
POLL = 180
STALE = 900  # 15 分钟无 last.pt 更新 = 判定死亡


def is_done():
    try:
        return json.load(open(STATUS)).get("stage") == "DONE"
    except Exception:
        return False


def latest_last_mtime():
    lasts = glob.glob(os.path.join(RUNS, "ext_yolo26n_*", "weights", "last.pt"))
    return max((os.path.getmtime(p) for p in lasts), default=0)


def gpu_busy():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout
        return len([l for l in out.splitlines() if l.strip()]) > 0
    except Exception:
        return False


def spawn(cmd, env_extra=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    subprocess.Popen(cmd, stdout=open(LOG, "ab"), stderr=subprocess.STDOUT,
                     env=env,
                     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP |
                                  getattr(subprocess, "DETACHED_PROCESS", 0))


def main():
    print("[supervise] start, POLL=%ds STALE=%ds" % (POLL, STALE), flush=True)
    while True:
        if is_done():
            print("[supervise] DONE detected -> exit", flush=True)
            break
        mt = latest_last_mtime()
        stale = (time.time() - mt) > STALE if mt else True
        if os.path.exists(FINAL) and stale and not gpu_busy():
            print("[supervise] final exists but not DONE -> rerun eval", flush=True)
            spawn([sys.executable, EVAL, FINAL])
        elif stale and not gpu_busy():
            print("[supervise] STALE + gpu free -> relaunch training(resume)", flush=True)
            spawn([sys.executable, TRAIN], {"YD_RESUME": "1"})
        else:
            print("[supervise] alive (last.pt fresh or gpu busy) -> wait", flush=True)
        time.sleep(POLL)


if __name__ == "__main__":
    main()
