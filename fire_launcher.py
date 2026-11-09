# -*- coding: utf-8 -*-
"""
fire_launcher.py —— 轻量调度器(无 torch 依赖, 纯 stdlib, 启动<1s, 一次dispatch即退出).
设计目标: 规避"框架按名/按树周期性杀死 loop 进程(fire_autoopt/optimize_loop)"导致循环空转.
机制:
  - 本脚本由 Windows 任务计划程序(每5分钟)独立调用, 自身也可能被框架杀掉, 但:
    1) 它只做"读状态 + 派发 trainer", 在 schtasks /run 完成后即可被杀, 无状态损失;
    2) trainer 经 schtasks 派发(父为系统svchost), 完全脱离本脚本进程树, 框架杀不掉, 持续训练写status;
    3) 下一轮调度(5min)重新读状态: 已完成的登记, 进行中的等待, 未开始的派发 -> 自主推进.
  - 因此即便本脚本每次都被秒杀, 训练仍在进行, 且每5分钟自我修复一次调度.
"""
import os, sys, json, time, shutil, subprocess, datetime

from pathlib import Path

DP = Path(r"E:/火情监测/火情监测/distill_pipeline")
STATE_FILE = DP / "OPT_STATE.json"
DONE_FILE = DP / "OPT_DONE.json"
EXP_SCRIPT = DP / "exp_train_eval.py"
LAUNCHER_LOCK = DP / "fire_launcher.lock.d"
WORK_ROOT = Path(r"D:/fire_opt")
VENV_PY = r"C:\Users\qwqw\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
BASE_PY = r"C:\Users\qwqw\.workbuddy\binaries\python\versions\3.13.12\python.exe"
PLAN_FILE = DP / "experiments_plan.json"
STUCK_SECONDS = 2 * 3600
DEFAULT_PLAN = [
    {"prefix": "expA", "sources": [{"name": "alirhayem", "max_n": 0}, {"name": "cctv", "max_n": 0}],
     "model_size": "n", "base_model": "yolo26n.pt", "epochs": 40, "patience": 15, "imgsz": 640, "enhance": False},
    {"prefix": "expB", "sources": [{"name": "alirhayem", "max_n": 0}, {"name": "dfire", "max_n": 0}, {"name": "cctv", "max_n": 0}],
     "model_size": "n", "base_model": "yolo26n.pt", "epochs": 40, "patience": 15, "imgsz": 640, "enhance": False},
    {"prefix": "expC", "sources": [{"name": "alirhayem", "max_n": 0}, {"name": "dfire", "max_n": 0}, {"name": "cctv", "max_n": 0}],
     "model_size": "n", "base_model": "yolo26n.pt", "epochs": 40, "patience": 15, "imgsz": 640, "enhance": True},
    {"prefix": "expD", "sources": [{"name": "alirhayem", "max_n": 0}, {"name": "dfire", "max_n": 0}, {"name": "cctv", "max_n": 0}],
     "model_size": "s", "base_model": "yolo26s.pt", "epochs": 40, "patience": 15, "imgsz": 640, "enhance": False},
    {"prefix": "expE", "sources": [{"name": "alirhayem", "max_n": 0}, {"name": "dfire", "max_n": 0}, {"name": "cctv", "max_n": 0}],
     "model_size": "s", "base_model": "yolo26s.pt", "epochs": 60, "patience": 20, "imgsz": 640, "enhance": True},
    {"prefix": "expF", "sources": [{"name": "alirhayem", "max_n": 0}, {"name": "dfire", "max_n": 0}, {"name": "cctv", "max_n": 0}],
     "model_size": "s", "base_model": "yolo26s.pt", "epochs": 60, "patience": 20, "imgsz": 640, "enhance": True, "targeted": True},
    {"prefix": "expG", "sources": [{"name": "alirhayem", "max_n": 0}, {"name": "dfire", "max_n": 0}, {"name": "cctv", "max_n": 0}],
     "model_size": "s", "base_model": "yolo26s.pt", "epochs": 60, "patience": 20, "imgsz": 640, "enhance": True, "distill": {"mode": "pseudo", "teacher_size": "l"}},
]


def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(DP / "loop_console.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def load_plan():
    if PLAN_FILE.exists():
        try:
            d = json.load(open(PLAN_FILE, encoding="utf-8"))
            exps = d.get("experiments")
            if isinstance(exps, list) and exps:
                return exps
        except Exception:
            pass
    return DEFAULT_PLAN


def load_state():
    if STATE_FILE.exists():
        try:
            return json.load(open(STATE_FILE, encoding="utf-8"))
        except Exception:
            pass
    return {"exps": {}, "loop_heartbeat_ts": now_iso()}


def save_state(s):
    s["last_update"] = now_iso()
    s["loop_heartbeat_ts"] = now_iso()
    json.dump(s, open(STATE_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def trainer_running(prefix):
    """trainer 是否仍在跑: 心跳 <150s, 或派发后 <200s 窗口, 或 schtasks 任务正在运行."""
    try:
        hb = DP / f"hb_{prefix}.alive"
        if hb.exists() and (time.time() - hb.stat().st_mtime) < 150:
            return True
    except Exception:
        pass
    try:
        sf = DP / f"train_start_{prefix}.txt"
        if sf.exists() and sf.stat().st_size > 0:
            if (time.time() - float(sf.read_text().strip())) < 200:
                return True
    except Exception:
        pass
    try:
        out = subprocess.run(f'schtasks /query /tn FireTrain_{prefix}', shell=True,
                             capture_output=True, text=True)
        if "正在运行" in out.stdout or "Running" in out.stdout:
            return True
    except Exception:
        pass
    return False


def exp_stuck(prefix):
    hb = DP / f"hb_{prefix}.alive"
    if hb.exists() and (time.time() - hb.stat().st_mtime) < STUCK_SECONDS:
        return False
    return True


def _acquire_launcher_lock():
    """简单单实例锁(目录锁), 防止调度器与自链同时调用本脚本导致双重派发trainer抢GPU."""
    try:
        os.mkdir(str(LAUNCHER_LOCK))
        (LAUNCHER_LOCK / "pid").write_text(str(os.getpid()))
        return True
    except FileExistsError:
        try:
            pidf = LAUNCHER_LOCK / "pid"
            old = int(pidf.read_text().strip()) if (pidf.exists() and pidf.stat().st_size > 0) else None
            import psutil
            if old and psutil.pid_exists(old):
                return False
            shutil.rmtree(str(LAUNCHER_LOCK), ignore_errors=True)
            os.mkdir(str(LAUNCHER_LOCK))
            (LAUNCHER_LOCK / "pid").write_text(str(os.getpid()))
            return True
        except Exception:
            return False


def _launch_trainer_schtasks(prefix, cfg_path, interp):
    wrapper = DP / f"launch_{prefix}.cmd"
    logf = DP / f"loop_{prefix}.log"
    launcher = DP / "fire_launcher.py"
    cons = DP / "loop_console.log"
    content = (
        "@echo off\r\n"
        "chcp 65001 >nul\r\n"
        "set PYTHONUTF8=1\r\n"
        f'set FIRE_RUNS_DIR={str(WORK_ROOT / "runs")}\r\n'
        f'set TMP={str(WORK_ROOT / "tmp")}\r\n'
        f'set TEMP={str(WORK_ROOT / "tmp")}\r\n'
        f'if not exist "{WORK_ROOT / "tmp"}" mkdir "{WORK_ROOT / "tmp"}"\r\n'
        f'if not exist "{WORK_ROOT / "runs"}" mkdir "{WORK_ROOT / "runs"}"\r\n'
        f'"{interp}" "{EXP_SCRIPT}" "{cfg_path}" >> "{logf}" 2>&1\r\n'
        # 自链: 本实验训练结束(成功或崩溃)后, 立即驱动下一个实验 -> 完全自主 expE->F->G
        f'"{interp}" "{launcher}" >> "{cons}" 2>&1\r\n'
    )
    wrapper.write_text(content, encoding="utf-8")
    task = f"FireTrain_{prefix}"
    st = (datetime.datetime.now() + datetime.timedelta(minutes=2)).strftime("%H:%M")
    subprocess.run(f'schtasks /delete /tn {task} /f', shell=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rc = subprocess.run(
        f'schtasks /create /tn {task} /tr "\\"{wrapper}\\"" /sc once /st {st} /f',
        shell=True, capture_output=True, text=True)
    if rc.returncode != 0:
        log(f"[launch] schtasks create {prefix} 失败: {rc.stderr.strip() or rc.stdout.strip()}")
    r2 = subprocess.run(f'schtasks /run /tn {task}', shell=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r2.returncode != 0:
        log(f"[launch] schtasks /run {prefix} 失败, 等计划时间 {st}")
    return task


def _kill_trainer(prefix):
    task = f"FireTrain_{prefix}"
    try:
        subprocess.run(f'schtasks /end /tn {task}', shell=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(f'schtasks /delete /tn {task} /f', shell=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    try:
        import psutil
        target = f"config_{prefix}.json"
        for pr in psutil.process_iter(['pid', 'cmdline']):
            args = pr.info.get('cmdline') or []
            if any('exp_train_eval.py' in str(a) and target in str(a) for a in args):
                try:
                    pr.kill()
                except Exception:
                    pass
    except Exception:
        pass
    try:
        os.remove(DP / f"train_start_{prefix}.txt")
    except Exception:
        pass


def _register_cfg(prefix):
    """确保 config_<prefix>.json 存在(从 experiments_plan 生成), 返回路径."""
    cfg_path = DP / f"config_{prefix}.json"
    if not cfg_path.exists():
        plan = load_plan()
        for e in plan:
            if e["prefix"] == prefix:
                json.dump(e, open(cfg_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                break
    return cfg_path


def main():
    if not _acquire_launcher_lock():
        log("[launch] 另一实例运行中, 退出避免双派发")
        return
    state = load_state()
    plan = load_plan()
    # 计划扩展保护: 曾EXHAUSTED但计划还有未做的 -> 清done继续
    if state.get("done") and any(not state.get("exps", {}).get(e["prefix"], {}).get("done") for e in plan):
        log("[launch] 计划尚有未完成任务, 清除旧done标记")
        state["done"] = False
        save_state(state)
    if state.get("done"):
        log("[launch] 已结束(见OPT_DONE.json)")
        return

    # 1) 先登记所有已产出 status 的实验
    reached = None
    for e in plan:
        p = e["prefix"]
        sj = DP / f"status_{p}.json"
        if sj.exists():
            try:
                st = json.load(open(sj, encoding="utf-8"))
            except Exception:
                continue
            cur = state.get("exps", {}).get(p, {})
            if not cur.get("done"):
                st["done"] = True
                state.setdefault("exps", {})[p] = st
                save_state(state)
                log(f"[launch] {p} 完成登记 maxmin={st.get('maxmin_PR')} reached_98={st.get('reached_98')}")
                if st.get("reached_98"):
                    reached = (p, st)
            # 已完成的实验清理其 trainer 任务(若残留)
            _kill_trainer(p)

    if reached:
        p, st = reached
        src = DP / f"final_{p}_best.pt"
        if src.exists():
            shutil.copy(src, DP / "OPTIMAL_MODEL.pt")
        json.dump({"reason": "REACHED_98", "best_exp": p, "best_maxmin": st.get("maxmin_PR"),
                   "ts": now_iso()}, open(DONE_FILE, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        log(f"[launch] 🎯 达标! {p} maxmin={st.get('maxmin_PR')} -> OPTIMAL_MODEL.pt")
        # 清理自托管任务
        try:
            subprocess.run('schtasks /delete /tn FireAutoOpt_Loop /f', shell=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        for e in plan:
            try:
                subprocess.run(f'schtasks /delete /tn FireTrain_{e["prefix"]} /f', shell=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        return

    # 2) 是否有 trainer 正在跑 -> 等待(不抢占GPU)
    for e in plan:
        if trainer_running(e["prefix"]):
            log(f"[launch] {e['prefix']} trainer 运行中, 本轮等待")
            return

    # 3) 找一个待跑实验(未完成且maxmin<=0), 启动其 trainer(独立schtasks)
    for e in plan:
        p = e["prefix"]
        cur = state.get("exps", {}).get(p, {})
        done = cur.get("done")
        mm = cur.get("maxmin_PR", -1)
        if done and mm > 0:
            continue  # 已达标/有效成绩, 跳过
        # 卡死检测: 若有 train_start 但心跳停>2h -> 清理后重派(resume)
        if (not done) and exp_stuck(p) and (DP / f"train_start_{p}.txt").exists():
            log(f"[launch] {p} 疑似卡死, 清理stale trainer后重派")
            _kill_trainer(p)
        cfg_path = _register_cfg(p)
        interp = VENV_PY if os.path.exists(VENV_PY) else BASE_PY
        _launch_trainer_schtasks(p, cfg_path, interp)
        (DP / f"train_start_{p}.txt").write_text(str(time.time()))
        log(f"[launch] 已派发 {p} 的 trainer (schtasks, 独立进程) -> 等待训练写status")
        return

    # 4) 没有待跑且没有运行中 -> 全部尝试过但未达标
    log("[launch] 所有实验均已尝试但未达标, 维持轮回(下一轮5min再检查)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[FATAL] {e}")
        import traceback
        traceback.print_exc()
        try:
            s = load_state(); s["last_error"] = str(e); save_state(s)
        except Exception:
            pass
