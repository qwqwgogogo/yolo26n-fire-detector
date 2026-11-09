# -*- coding: utf-8 -*-
"""
optimize_loop.py —— 模型自主优化循环主控 (用户铁目标: 官方1100绝不进训练, 图像级P/R冲到98%才停)
流程:
  1) 等待 Exp1(status_ext.json, 由 train_external.py 产出)
  2) 顺序跑 EXPERIMENTS (exp2..exp7, 逐步放大数据量/增强/扩容yolo26s)
  3) 每个实验: 构建合并公开集 -> 训练 -> 官方1100无泄漏评估
  4) 任一实验达到 P>=0.98 且 R>=0.98 -> 复制 OPTIMAL_MODEL.pt 并写 OPT_DONE.json 停止
  5) 全部用尽仍未达 -> 写 OPT_DONE.json(EXHAUSTED, 记录最佳maxmin)
健壮性: 状态持久化(OPT_STATE.json, 每60s心跳) + 顶层try/except + 单实验失败不阻断循环
看门狗: HOURLY automation 依据 loop_heartbeat_ts 判断卡死/崩溃并重启本脚本
"""
import os, sys, json, time, shutil, subprocess, datetime, atexit
from pathlib import Path

# 中文路径修复: 本项目目录含中文(火情监测), 在 Windows 非UTF8代码页(默认GBK)下,
# subprocess 启动含中文路径的脚本子进程会报 "Unable to create process ... E:\????\????" -> 评估子进程起不来
# -> 实验永远写不出 status -> 循环"max tries exceeded, no status"空转(已踩坑: expE/F/G 全部无成绩).
# 强制 UTF-8 模式让 Python 用宽字符 API 启动子进程, 彻底解决. 启动命令也应带 PYTHONUTF8=1.
os.environ.setdefault("PYTHONUTF8", "1")

# 自修正: 后台任务框架有时用 base python(无 torch/ultralytics)启动本脚本 -> 子进程 ImportError 崩(rc=1).
# 若当前解释器导入不了 torch, 自动用 venv python 重新执行本脚本(os.execv 替换进程映像, PID不变),
# 这样无论框架用 base 还是 venv python 拉起, 最终都跑在带依赖的 venv 上. 必须在其它 import 前做.
VENV_PY = r"C:\Users\qwqw\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
try:
    import torch  # noqa: F401
except Exception:
    if os.path.exists(VENV_PY) and os.path.abspath(VENV_PY) != os.path.abspath(sys.executable):
        try:
            os.execv(VENV_PY, [VENV_PY, os.path.abspath(__file__)] + sys.argv[1:])
        except Exception:
            pass
    # 若 venv 也缺失, 继续往下跑(会在训练子进程处报错, 有日志可查)

ROOT = Path(r"E:/火情监测/火情监测")
DP = ROOT / "distill_pipeline"
STATE_FILE = DP / "OPT_STATE.json"
DONE_FILE = DP / "OPT_DONE.json"
EXP_SCRIPT = DP / "exp_train_eval.py"
MAX_WAIT_EXP1 = 720  # 720*60s = 12h
EXP_TIMEOUT = 40 * 3600   # 单实验硬超时40h(兜底防OOM卡死假活), 到点kill后resume重试
MAX_TRIES_PER_EXP = 12    # 单实验最多连续重试次数(resume续训); 仍失败则标记失败, 后续轮回再试
MIN_RAM_GB = 3.0          # 启动训练前所需最小空闲内存; 不足则等待(持续心跳), 避免一启动就崩
MAX_WAIT_RAM_MIN = 180    # 单实验等待内存的最长时间(分钟); 超时则直接尝试(resume可恢复)
STUCK_SECONDS = 2 * 3600  # 训练输出2h无更新即判卡死(敏锐发现OOM挂起)
STUCK_WARMUP = 20 * 60    # 启动宽限期: 前20min不判卡死(让数据集构建+前几个epoch完成, 避免误杀)
WORK_ROOT = Path(r"D:/fire_opt")   # 训练产物根(迁到D盘, 避开E盘99%满导致的写盘崩溃)
LOCK_DIR = DP / "optimize_loop.lock.d"  # 单实例锁(目录锁, mkdir在Windows上原子可靠, 比O_EXCL文件锁稳)
TRAINER_LOCK = DP / "trainer.lock.d"     # 全局训练互斥锁: 同一时刻只允许一个 exp_train_eval 在跑
                                           # (防框架反复重生多个loop实例时, 多trainer抢GPU互杀 rc=15 空转)


def _release_lock_if_holder():
    """仅当本实例是当前锁持有者时才删除锁目录(atexit 调用).
    防止'被接管的僵尸实例'退出时误删真正持有者的锁."""
    try:
        if LOCK_DIR.exists():
            pidf = LOCK_DIR / "pid"
            if pidf.exists() and pidf.stat().st_size > 0:
                try:
                    if int(pidf.read_text().strip()) == os.getpid():
                        shutil.rmtree(str(LOCK_DIR), ignore_errors=True)
                except Exception:
                    pass
    except Exception:
        pass


def verify_still_holder():
    """周期性确认本实例仍是锁目录记录的持有者. 若锁被其他实例接管(本实例沦为僵尸),
    返回 False -> 调用方应优雅退出, 避免僵尸循环与持有者互杀训练子进程(rc=15)空转."""
    try:
        pidf = LOCK_DIR / "pid"
        if not (pidf.exists() and pidf.stat().st_size > 0):
            return False
        cur = int(pidf.read_text().strip())
        if cur != os.getpid():
            log(f"[singleton] 锁已被其他实例(pid={cur})接管, 本实例(pid={os.getpid()})退出")
            return False
    except Exception:
        return False
    return True


def _release_trainer_lock_if_holder():
    """仅当本实例是当前训练锁持有者时才删除锁目录(atexit/训练结束后调用)."""
    try:
        if TRAINER_LOCK.exists():
            pidf = TRAINER_LOCK / "pid"
            if pidf.exists() and pidf.stat().st_size > 0:
                if int(pidf.read_text().strip()) == os.getpid():
                    shutil.rmtree(str(TRAINER_LOCK), ignore_errors=True)
    except Exception:
        pass


def acquire_trainer_lock():
    """全局训练互斥锁: 同一时刻只允许一个 exp_train_eval 在跑(防多loop实例抢GPU互杀 rc=15).
    目录锁+mkdir原子; 含pid, 持有者崩溃则被接管(避免死锁导致永远无法训练)."""
    try:
        os.mkdir(str(TRAINER_LOCK))
        (TRAINER_LOCK / "pid").write_text(str(os.getpid()))
        atexit.register(_release_trainer_lock_if_holder)
        return True
    except FileExistsError:
        pass
    try:
        pidf = TRAINER_LOCK / "pid"
        old = int(pidf.read_text().strip()) if (pidf.exists() and pidf.stat().st_size > 0) else None
        import psutil
        if old and psutil.pid_exists(old):
            return False  # 别人正在训练, 本实例(多半是非持有者)跳过
        shutil.rmtree(str(TRAINER_LOCK), ignore_errors=True)
        os.mkdir(str(TRAINER_LOCK))
        (TRAINER_LOCK / "pid").write_text(str(os.getpid()))
        atexit.register(_release_trainer_lock_if_holder)
        return True
    except Exception:
        return False


def acquire_singleton():
    """单实例锁(目录锁, Windows上最可靠): os.mkdir 在同名目录已存在时原子地抛 FileExistsError,
    谁先抢到谁持有, 其余实例立即退出. 此前用 os.open(O_EXCL) 文件锁在本机 Python/NTFS 上不可靠
    (多余实例未拿到报错也未退出 -> 进程群). 多实例会互相踩GPU/OPT_STATE导致空转, 故必须只留一个.
    双保险: 即便抢到锁, 若存在更早启动的兄弟实例(框架反复重启派生), 本实例是后来者 -> 立即退出,
    绝不抢GPU/ spawning trainer, 从源头消除"双循环互杀对方训练子进程(rc=15)"的空转."""
    import shutil
    got = False
    try:
        os.mkdir(str(LOCK_DIR))            # 原子: 仅一个进程成功
        got = True
    except FileExistsError:
        got = False
    if got:
        try:
            (LOCK_DIR / "pid").write_text(str(os.getpid()))
        except Exception:
            pass
        atexit.register(_release_lock_if_holder)
    else:
        # 锁目录已存在 -> 判定持有者是否存活(无pid文件/空锁目录=残留死锁, 直接接管)
        try:
            pidf = LOCK_DIR / "pid"
            old_pid = int(pidf.read_text().strip()) if (pidf.exists() and pidf.stat().st_size > 0) else None
        except Exception:
            old_pid = None
        try:
            import psutil
            holder_alive = bool(old_pid and old_pid != os.getpid() and psutil.pid_exists(old_pid))
        except Exception:
            holder_alive = False
        if holder_alive:
            log(f"[singleton] 已有实例 pid={old_pid} 持有锁, 本实例(pid={os.getpid()})退出避免多开")
            sys.exit(0)
        # 持有者不存在/已死/无pid -> 接管(清洗残留空锁目录后重新加锁)
        try:
            shutil.rmtree(str(LOCK_DIR), ignore_errors=True)
        except Exception:
            pass
        try:
            os.mkdir(str(LOCK_DIR)); got = True
            (LOCK_DIR / "pid").write_text(str(os.getpid()))
            atexit.register(_release_lock_if_holder)
        except Exception:
            pass
    # 锁目录是唯一种裁者: 仅 mkdir 成功的实例成为持有者并 spawn trainer;
    # 任何框架/看门狗重生的副本 mkdir 必失败且持有者存活 -> 直接退出, 不会抢GPU/双开.
    # (此前加的"更早兄弟实例"时间判断过于激进, 会误杀唯一合法实例, 已移除)
    log(f"[singleton] 本实例(pid={os.getpid()}) 取得唯一控制权")


_LAST_CULL = 0.0
def cull_other_loops():
    """防御性清场: 主动杀掉其他 optimize_loop.py 实例. 外部(看门狗/后台任务框架)可能在进程死后
    不断重生循环形成级联(38716->36280->18844 链条), 多实例抢GPU/OPT_STATE会致空转. 只有持有锁的
    本实例是真正干活的; 其余兄弟一律清除. 不清自己/祖先/自己的训练子进程(exp_train_eval)."""
    try:
        import psutil
        me = os.getpid()
        anc = set()
        p = psutil.Process(me)
        while p:
            anc.add(p.pid); p = p.parent()
        # 当前锁持有者 pid: 竞争loop只清"非持有者", 绝不误杀持有者(否则双loop互杀)
        holder_pid = None
        try:
            pidf = LOCK_DIR / "pid"
            if pidf.exists() and pidf.stat().st_size > 0:
                holder_pid = int(pidf.read_text().strip())
        except Exception:
            holder_pid = None
        # 防御: 若本实例即是持有者(锁pid==me)但读取异常, 用 me 兜底, 绝不错杀自己的trainer
        effective_holder = holder_pid if holder_pid is not None else me
        killed = []
        for pr in psutil.process_iter(['pid', 'cmdline']):
            pid = pr.info['pid']
            if pid == me or pid in anc:
                continue
            args = pr.info.get('cmdline') or []
            # 精确匹配: 命令行参数以脚本名结尾, 避免误杀"源码里含该字符串"的监控脚本
            is_loop = any(str(a).endswith('optimize_loop.py') for a in args)
            is_train = any(str(a).endswith('exp_train_eval.py') for a in args)
            if is_loop and not is_train:
                # 其他优化循环: 仅当它不是当前锁持有者时才清除(持有者必须保留!)
                if pid == holder_pid:
                    continue
                try:
                    pr.kill(); killed.append(pid)
                except Exception:
                    pass
            elif is_train:
                # 训练子进程: 只保留"当前锁持有者(effective_holder)"自己的 trainer; 其余一概清除.
                # 包括: 竞争loop派生的trainer、框架重生的孤儿trainer、父不是loop的残留trainer.
                # 这样无论双开还是重生, 最终全局只有持有者的一个trainer在用GPU ->
                # 杜绝双trainer抢GPU致OOM互杀(rc=15)空转. 持有者自身trainer的parent==effective_holder, 不会被杀.
                parent = pr.ppid()
                if parent != effective_holder and parent != me:
                    try:
                        pr.kill(); killed.append(pid)
                    except Exception:
                        pass
        if killed:
            log(f"[cull] 清除竞争实例: {killed}")
    except Exception as e:
        log(f"[cull] err {e}")


def exp_stuck(prefix):
    """检测实验是否卡死: 仅看心跳文件 hb_<prefix>.alive 是否在 STUCK_SECONDS 内刷新.
    exp_train_eval 内部守护线程每30s刷新心跳(覆盖 构建/训练/评估 全阶段),
    故评估阶段(不更新results.csv)不会被误判卡死; 真正挂起(冻死/OOM swap)则心跳停 -> 判卡死.
    注意: 必须用心跳而非results.csv, 否则评估阶段(约数十秒~数分钟无results更新)+个别慢epoch
    会被误杀, 导致评估永远写不出status、循环空转(本项目已踩坑)."""
    now = time.time()
    hb = DP / f"hb_{prefix}.alive"
    if hb.exists() and (now - hb.stat().st_mtime) < STUCK_SECONDS:
        return False
    return True


def avail_phys_gb():
    """读取当前物理空闲内存(GB)"""
    try:
        import ctypes
        class M(ctypes.Structure):
            _fields_ = [("d", ctypes.c_ulong), ("l", ctypes.c_ulong),
                        ("t", ctypes.c_ulonglong), ("a", ctypes.c_ulonglong),
                        ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                        ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong),
                        ("ae", ctypes.c_ulonglong)]
        m = M(); m.d = ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.a / 1e9
    except Exception:
        return 99.0


def ram_ok():
    try:
        return avail_phys_gb() >= MIN_RAM_GB
    except Exception:
        return True

# 实验计划: 从 experiments_plan.json 加载(便于增删实验而无需改代码).
# 三条并行提分产线: A宽数据训练(expA-E) / B蒸馏(expG) / C针对性样本(expF)
# 数据均来自公开集, 官方1100在 build_dataset 里被文件名兜底排除
PLAN_FILE = DP / "experiments_plan.json"
# 兜底默认(与JSON一致, 防止JSON缺失)
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


def load_plan():
    if PLAN_FILE.exists():
        try:
            d = json.load(open(PLAN_FILE, encoding="utf-8"))
            exps = d.get("experiments")
            if isinstance(exps, list) and exps:
                return exps
        except Exception as e:
            log(f"[plan] 读 experiments_plan.json 失败 {e}, 用默认")
    return DEFAULT_PLAN


def plan_done(state, plan):
    """计划里所有实验是否都已完成"""
    done = state.get("exps", {})
    return all(done.get(e["prefix"], {}).get("done") for e in plan)


def pending_plan(state, plan):
    """计划里尚未完成的实验"""
    done = state.get("exps", {})
    return [e for e in plan if not done.get(e["prefix"], {}).get("done")]


def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


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
    # 每60s 周期性清兄弟实例(兜底框架持续重生) + 确认本实例仍是锁持有者(被接管则退出)
    global _LAST_CULL
    t = time.time()
    if t - _LAST_CULL > 60:
        _LAST_CULL = t
        try:
            cull_other_loops()
        except Exception:
            pass
        try:
            if not verify_still_holder():
                sys.exit(0)
        except Exception:
            pass


def write_done(reason, best_exp=None, best_maxmin=None):
    d = {"reason": reason, "best_exp": best_exp, "best_maxmin": best_maxmin, "ts": now_iso()}
    json.dump(d, open(DONE_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log(f"[done] {d}")


def wait_exp1(state):
    log("[loop] 等待 Exp1(status_ext.json)... 当前由 train_external.py 独立训练")
    sj = DP / "status_ext.json"
    for _ in range(MAX_WAIT_EXP1):
        if sj.exists():
            try:
                st = json.load(open(sj, encoding="utf-8"))
                P = float(st.get("P", 0)); R = float(st.get("R", 0))
                st["maxmin_PR"] = min(P, R)
                st["reached_98"] = (min(P, R) >= 0.98)
                state["exp1"] = st; state["exp1_done"] = True
                save_state(state)
                log(f"[loop] Exp1 完成: P={P} R={R} maxmin={min(P, R):.4f}")
                return True
            except Exception as e:
                log(f"[loop] 读status_ext失败 {e}")
        save_state(state)
        time.sleep(60)
    log("[loop] Exp1 等待超时(12h), 标记失败继续(后续实验会用公开集重训等价模型)")
    state["exp1"] = {"failed": True}; state["exp1_done"] = True
    save_state(state)
    return False


def run_exp(cfg, state):
    prefix = cfg["prefix"]
    # 非持有者(被框架重生的竞争实例)立即退出本实验, 绝不抢GPU/启动trainer -> 杜绝双trainer互杀
    if not verify_still_holder():
        log(f"[loop] {prefix} 本实例非锁持有者, 跳过训练(避免多开)")
        return {"prefix": prefix, "skipped": True, "reached_98": False, "maxmin_PR": -1, "rc": 0}
    # 全局训练互斥锁: 整个实验期间只持有一个trainer(防多loop抢GPU互杀). 抢不到则跳过(单实例下不会发生).
    if not acquire_trainer_lock():
        log(f"[loop] {prefix} 训练锁被其他实例持有, 跳过(全局仅一个trainer)")
        return {"prefix": prefix, "skipped": True, "reached_98": False, "maxmin_PR": -1, "rc": 0}
    try:
        cfg_path = DP / f"config_{prefix}.json"
        json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        # 产物迁到D盘(避开E盘99%满), 同时把TMP/TEMP也指到D盘, 彻底绕开E盘写盘崩溃
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"   # 子进程(含中文路径的exp_train_eval/tiled_eval)必须UTF-8模式
        env["FIRE_RUNS_DIR"] = str(WORK_ROOT / "runs")
        env["TMP"] = str(WORK_ROOT / "tmp"); env["TEMP"] = str(WORK_ROOT / "tmp")
        os.makedirs(env["TMP"], exist_ok=True)
        os.makedirs(WORK_ROOT / "runs", exist_ok=True)
        for attempt in range(1, MAX_TRIES_PER_EXP + 1):
            # 内存预条件: 空闲不足则等待(持续心跳), 避免一启动就因抢内存被系统杀
            waited = 0
            while not ram_ok():
                if waited == 0:
                    log(f"[loop] {prefix} 空闲内存不足({avail_phys_gb():.1f}GB<{MIN_RAM_GB}GB), 等待释放...")
                save_state(state); time.sleep(60); waited += 1
                if waited >= MAX_WAIT_RAM_MIN * 60:
                    log(f"[loop] {prefix} 等内存超时({MAX_WAIT_RAM_MIN}min), 直接尝试(resume可恢复)")
                    break
            log(f"[loop] 启动 {prefix} (attempt {attempt}/{MAX_TRIES_PER_EXP})")
            # 输出落到日志文件(不再丢进DEVNULL), 评估报错可排查; 心跳由子进程守护线程刷新
            logf = DP / f"loop_{prefix}.log"
            # 子进程解释器优先用 venv(带 torch/ultralytics); 兜底用当前解释器
            interp = VENV_PY if os.path.exists(VENV_PY) else sys.executable
            with open(logf, "a", encoding="utf-8") as lf:
                proc = subprocess.Popen([interp, str(EXP_SCRIPT), str(cfg_path)],
                                        stdout=lf, stderr=subprocess.STDOUT, env=env)
            launch_ts = time.time()
            deadline = launch_ts + EXP_TIMEOUT
            while proc.poll() is None:
                elapsed = time.time() - launch_ts
                # 卡死检测: 仅在宽限期后生效(避免数据集构建阶段被误杀); OOM挂起或硬超时 -> kill后resume
                stuck = elapsed > STUCK_WARMUP and exp_stuck(prefix)
                if stuck or time.time() > deadline:
                    reason = "卡死检测" if stuck else f"硬超时{EXP_TIMEOUT//3600}h"
                    log(f"[loop] {prefix} attempt{attempt} {reason}, kill 子进程准备resume重试")
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    try:
                        proc.wait(timeout=60)
                    except Exception:
                        pass
                    break
                save_state(state)   # 持续心跳, 防看门狗误判
                time.sleep(60)
            rc = proc.poll()
            if rc is None:
                continue  # 被kill, 进入下一attempt(resume续训)
            log(f"[loop] {prefix} attempt{attempt} 进程结束 rc={rc}")
            sj = DP / f"status_{prefix}.json"
            if sj.exists():
                try:
                    st = json.load(open(sj, encoding="utf-8"))
                    st["rc"] = rc
                    return st
                except Exception as e:
                    return {"prefix": prefix, "error": str(e), "reached_98": False, "maxmin_PR": -1, "rc": rc}
            else:
                log(f"[loop] {prefix} attempt{attempt} 结束但无status(rc={rc}); 若有last.pt则下次resume重试")
        return {"prefix": prefix, "error": f"max tries({MAX_TRIES_PER_EXP}) exceeded, no status",
                "reached_98": False, "maxmin_PR": -1, "rc": -1}
    finally:
        _release_trainer_lock_if_holder()  # 实验结束(成功/失败/退出)才释放训练锁


def main():
    global _LAST_CULL
    acquire_singleton()   # 单实例锁: 防止多循环抢GPU/状态(已踩坑)
    cull_other_loops()    # 启动即清掉任何已存在的兄弟实例(框架重生的级联)
    state = load_state()
    plan = load_plan()
    # 计划扩展保护: 若上次EXHAUSTED但计划里还有未做的实验(如新增expF/G), 清除done继续
    if state.get("done") and pending_plan(state, plan):
        log("[loop] 检测到计划尚有未完成任务, 清除旧的done标记继续")
        state["done"] = False
        save_state(state)
    if state.get("done"):
        log("[loop] 已结束(见OPT_DONE.json)"); return
    if not state.get("exp1_done"):
        wait_exp1(state)
    cycle = 0
    while True:   # 永不停(用户: 不达标不停止); 任一达标即return; 否则轮回冲击
        # 被接管的僵尸实例在此优雅退出(持有者继续), 杜绝双loop互杀训练子进程
        if not verify_still_holder():
            sys.exit(0)
        # 周期性清兄弟实例(兜底框架持续重生, 20s一次)
        tnow = time.time()
        if tnow - _LAST_CULL > 20:
            _LAST_CULL = tnow
            try:
                cull_other_loops()
            except Exception:
                pass
        cycle += 1
        log(f"[loop] ===== 第 {cycle} 轮回合 =====")
        progressed = False
        for cfg in plan:
            p = cfg["prefix"]
            st0 = state.get("exps", {}).get(p)
            # 已完成且产出有效成绩(含已达标)则跳过; 失败(无成绩)的才重跑
            if st0 and st0.get("done") and st0.get("maxmin_PR", -1) > 0:
                log(f"[loop] {p} 已完成(历史maxmin={st0.get('maxmin_PR'):.4f}), 跳过")
                continue
            log(f"[loop] === 运行 {p} ===")
            st = run_exp(cfg, state)
            st["done"] = True
            state.setdefault("exps", {})[p] = st
            save_state(state)
            if st.get("reached_98"):
                src = DP / f"final_{p}_best.pt"
                if src.exists():
                    shutil.copy(src, DP / "OPTIMAL_MODEL.pt")
                state["done"] = True
                save_state(state)
                write_done("REACHED_98", p, st.get("maxmin_PR"))
                log(f"[loop] 🎯 达标! {p} maxmin={st.get('maxmin_PR'):.4f} -> OPTIMAL_MODEL.pt")
                return
            progressed = True
        best = max([state.get("exps", {}).get(p, {}).get("maxmin_PR", -1)
                    for p in state.get("exps", {})], default=-1)
        if progressed:
            log(f"[loop] 本轮完成, 当前最佳 maxmin={best:.4f}, 未达0.98 -> 进入下一轮回合(持续冲击)")
        else:
            log(f"[loop] 本轮无新进展(全部已有成绩未达标), 最佳 maxmin={best:.4f}, 休眠后重试")
        time.sleep(300)  # 轮回间隔, 给系统喘息 + 等待内存/磁盘缓解


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[FATAL] {e}")
        import traceback
        traceback.print_exc()
        try:
            s = load_state(); s["last_error"] = str(e)
            save_state(s)
        except Exception:
            pass
        sys.exit(1)
