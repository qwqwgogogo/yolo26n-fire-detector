# -*- coding: utf-8 -*-
"""
supervise.py —— 蒸馏流水线监督器（无人值守）。

- 若 run_distill.py 崩溃/退出非0/卡死, 自动重启(最多 MAX_RESTART 次)。
- 每 2 分钟记录一次 GPU 利用率与显存占用。
- 若 run_distill.log 超过 STALE_MIN 分钟无更新(疑似卡死), 杀掉重启。
- 检测到 status.json stage=DONE 即退出。
"""
import subprocess, time, sys, os, json
from pathlib import Path

# 限制线程数, 降低子进程(torch)内存提交, 避免小页面文件下 import 失败(WinError 1114/1455)
for _k, _v in {
    "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
    "VECLIB_MAXIMUM_THREADS": "4",
}.items():
    os.environ.setdefault(_k, _v)

ROOT = Path(r"E:/火情监测/火情监测")
PIPE = ROOT / "distill_pipeline"
PY = r"C:/Users/qwqw/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
SCRIPT = PIPE / "run_distill.py"
STATUS = PIPE / "status.json"
LOG = PIPE / "supervise.log"
RDLOG = PIPE / "run_distill.log"

MAX_RESTART = 12
STALE_MIN = 120  # 单次 250-epoch 训练约 1.8h, 必须 > 训练时长, 否则误杀
POLL_SEC = 90
MEM_LOW_MB = 600      # 可用物理内存低于此值视为危险, 主动降 batch 重启(防系统RAM耗尽崩)
MEM_BATCH_STEP = 8    # 每次降 batch 的步长


def latest_results_mtime():
    """返回 runs 目录下最新 results.csv 的修改时间(训练每 epoch 更新一次, 作心跳备份)。"""
    best = 0.0
    try:
        for p in Path(PIPE / "runs").rglob("results.csv"):
            try:
                mt = p.stat().st_mtime
                if mt > best:
                    best = mt
            except Exception:
                pass
    except Exception:
        pass
    return best


def log(s):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + str(s)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def is_done():
    if STATUS.exists():
        try:
            d = json.load(open(STATUS, encoding="utf-8"))
            return d.get("stage") == "DONE"
        except Exception:
            return False
    return False


def gpu_snapshot():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip().replace("\n", " | ")
    except Exception as e:
        return "nvidia-smi err: " + repr(e)


def avail_phys_mb():
    """用 ctypes 直接读系统可用物理内存(绕过被禁的 wmic 与偶发空输出的 PowerShell)。"""
    try:
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        ms = MEMORYSTATUSEX()
        ms.dwLength = ctypes.sizeof(ms)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return ms.ullAvailPhys / (1024.0 * 1024.0)
    except Exception:
        return -1.0


def cur_batch_n():
    try:
        return int(os.environ.get("YD_BATCH_N", "32"))
    except Exception:
        return 32


def main():
    log("supervisor start")
    for i in range(MAX_RESTART):
        if is_done():
            log("DONE detected -> exit supervisor")
            return
        log(f"=== supervisor launch #{i} ===")
        try:
            p = subprocess.Popen([PY, str(SCRIPT)], cwd=str(ROOT))
        except Exception as e:
            log("spawn error: " + repr(e))
            time.sleep(30)
            continue

        last_mtime = time.time()
        while p.poll() is None:
            if is_done():
                log("DONE detected while running -> stop supervisor")
                p.wait()
                return
            time.sleep(POLL_SEC)
            # GPU + 内存 快照(实时关注内存, 保证利用率最大化且不崩)
            mem = avail_phys_mb()
            log(f"GPU: {gpu_snapshot()} | MEM avail={mem:.0f}MB (batch_n={cur_batch_n()})")
            # 内存过低: 主动降 batch 重启, 防系统RAM耗尽崩(上次 epoch160 崩因)
            if mem > 0 and mem < MEM_LOW_MB and cur_batch_n() > 8:
                newb = max(8, cur_batch_n() - MEM_BATCH_STEP)
                os.environ["YD_BATCH_N"] = str(newb)
                log(f"MEM LOW ({mem:.0f}MB) -> lower YD_BATCH_N={newb}, kill & restart")
                p.kill()
                break
            # 卡死检测: run_distill.log 或 任一 results.csv(训练每 epoch 更新) 任一近期有活动即视为存活
            try:
                mt_log = os.path.getmtime(RDLOG)
            except Exception:
                mt_log = last_mtime
            mt_csv = latest_results_mtime()
            mt = max(mt_log, mt_csv, last_mtime)
            if time.time() - mt > STALE_MIN * 60:
                log(f"STALE > {STALE_MIN}m, killing pid {p.pid}")
                p.kill()
                break
            last_mtime = mt

        rc = p.poll()
        if rc is None:
            # 被 kill 打断(stale)
            log("process killed (stale) -> restart")
            time.sleep(15)
            continue
        if is_done():
            log(f"DONE detected (rc={rc}) -> exit supervisor")
            return
        # 未 DONE: 无论退出码如何都重启(受 MAX_RESTART 限制)
        log(f"run_distill exited rc={rc} 但非 DONE -> 20s 后重启")
        time.sleep(20)

    log("supervisor exhausted restarts -> end")


if __name__ == "__main__":
    main()
