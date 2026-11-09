#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载所有「免登录、可直接获取」的真实火情/火焰/烟雾数据集到 external_datasets/。
- GitHub 仓库用 git clone --depth 1（省空间）
- Google Drive 用 gdown
- HuggingFace 用 huggingface_hub.snapshot_download
- 每个数据源独立执行，单个失败不影响其他
- 磁盘守卫：剩余空间低于阈值(10GB) 即停止后续下载，避免撑爆 E 盘
"""
import os, sys, subprocess, shutil, time, datetime

BASE = "E:/火情监测/火情监测/external_datasets"
os.makedirs(BASE, exist_ok=True)
LOG = os.path.join(BASE, "download_log.txt")
GDOWN = "C:/Users/qwqw/.workbuddy/binaries/python/versions/3.13.12/Scripts/gdown.exe"
PY = "C:/Users/qwqw/.workbuddy/binaries/python/versions/3.13.12/python.exe"
FREE_THRESHOLD_GB = 10.0

GIT_REPOS = [
    # (本地目录名, git url, 备注)
    ("yuanjun_nav", "https://github.com/yuanjun0416/flames-and-smoke-datasets", "社区导航汇总(小)"),
    ("DBA-Fire", "https://github.com/ScryAbu/DBA-YOLO-Dataset", "3905图 YOLO"),
    ("DeepQuestAI_Fire-Smoke", "https://github.com/DeepQuestAI/Fire-Smoke-Dataset", "3000图 分类"),
    ("D-Fire", "https://github.com/gaiasd/DFireDataset", "21527图 YOLO 首选"),
    ("DFS", "https://github.com/siyuanwu/DFS-FIRE-SMOKE-Dataset", "9462图 YOLO/COCO"),
    ("FireNet", "https://github.com/OlafenwaMoses/FireNet", "62视频+160图 YOLO"),
    ("AIDER", "https://github.com/ckyrkou/AIDER", "无人机航拍"),
    ("FireDetn", "https://github.com/SuperXxts/FireDetn", "火检测"),
    ("FireSpot", "https://github.com/Biometrix-4/FireSpot-CNX", "火点检测"),
    ("CAIR", "https://github.com/cair/Fire-Detection-Image-Dataset", "分类(刻意不平衡)"),
    ("FLAME-SD", "https://github.com/AIS-Clemson/FLAME-SD", "航拍火(含分割)"),
    ("BA-UAV", "https://github.com/ipleiria-ciic/ees-datalab", "无人机火/烟"),
    ("firecam", "https://github.com/open-climate-tech/firecam", "烟雾相机"),
    ("FiSmo", "https://github.com/mtcazzolato/dsw2017", "9448图+158视频"),
    ("PYRONEAR-2025", "https://github.com/joseg20/wildfires2025", "15万标注/50k图 大"),
    ("gengyanlei", "https://github.com/gengyanlei/fire-smoke-detect-yolov4", "10k+图 VOC/XML(部分百度盘)"),
    ("huyquang_fire-smoke", "https://github.com/huyquang-bka/fire-smoke", "10万+图 YOLOv5 最大"),
    # bitbucket 用 get zip
    ("BoWFire", "bitbucket:gbdi/bowfire-dataset", "466图 城市/工业"),
]

GDOWN_FILES = [
    ("FireSmoke-Dataset_Txinyan", "1ddO1oVfL1i08FkAbp9mSLwuQDC6q41XW", "12032图 YOLO(GoogleDrive)"),
]

HF_REPOS = [
    ("MultiNatSmoke", "hongjinzhao0615/MultiNatSmoke", "11国烟雾聚合 分割(WACV2026)"),
]

def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def free_gb():
    try:
        u = shutil.disk_usage("E:/")
        return u.free / 1e9
    except Exception:
        return 999

def run(cmd, timeout=1800):
    try:
        r = subprocess.run(cmd, cwd=BASE, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr)[-1500:]
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"
    except Exception as e:
        return 1, str(e)

def guard():
    fg = free_gb()
    if fg < FREE_THRESHOLD_GB:
        log(f"!! 磁盘剩余 {fg:.1f}GB < {FREE_THRESHOLD_GB}GB 阈值，停止后续下载以避免撑爆")
        return False
    return True

def clone_git(name, url, note):
    log(f">>> 开始 [{name}] ({note})  free={free_gb():.1f}GB")
    dest = os.path.join(BASE, name)
    if os.path.exists(dest) and os.listdir(dest):
        log(f"    已存在，跳过: {dest}")
        return
    if url.startswith("bitbucket:"):
        user_repo = url.split(":", 1)[1]
        zipurl = f"https://bitbucket.org/{user_repo}/get/master.zip"
        code, out = run(["curl", "-L", "--max-time", "600", "-o", f"{name}.zip", zipurl])
        if code == 0 and os.path.exists(os.path.join(BASE, f"{name}.zip")):
            code2, out2 = run(["powershell", "-Command", f"Expand-Archive -Force '{name}.zip' '{name}'"])
            if code2 == 0:
                log(f"    ✓ bitbucket zip 解压完成")
            else:
                log(f"    ✗ 解压失败: {out2}")
        else:
            log(f"    ✗ bitbucket 下载失败: {out}")
        return
    code, out = run(["git", "clone", "--depth", "1", url, name], timeout=2400)
    if code == 0:
        log(f"    ✓ 完成: {name}")
    else:
        log(f"    ✗ 失败(code={code}): {out}")

def gdrivedl(name, fid, note):
    log(f">>> 开始 [{name}] ({note})  free={free_gb():.1f}GB")
    dest = os.path.join(BASE, name)
    os.makedirs(dest, exist_ok=True)
    if os.listdir(dest):
        log(f"    已存在，跳过")
        return
    code, out = run([GDOWN, fid, "-O", os.path.join(dest, f"{name}.zip")], timeout=1800)
    if code == 0:
        # 解压
        zipp = os.path.join(dest, f"{name}.zip")
        if os.path.exists(zipp):
            run(["powershell", "-Command", f"Expand-Archive -Force '{zipp}' '{dest}'"])
            log(f"    ✓ 完成(已解压): {name}")
        else:
            log(f"    ✓ 下载完成(未见zip，可能直接落盘): {name}")
    else:
        log(f"    ✗ 失败(code={code}): {out}")

def hfdl(name, repo, note):
    log(f">>> 开始 [{name}] ({note})  free={free_gb():.1f}GB")
    dest = os.path.join(BASE, name)
    if os.listdir(dest) if os.path.exists(dest) else False:
        log(f"    已存在，跳过")
        return
    code, out = run([PY, "-c",
        f"from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id='{repo}', local_dir=r'{dest}', max_workers=4)"],
        timeout=3600)
    if code == 0:
        log(f"    ✓ 完成: {name}")
    else:
        log(f"    ✗ 失败(code={code}): {out}")

def main():
    log("=" * 60)
    log("开始下载火情数据集  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    log(f"目标目录: {BASE}  磁盘阈值: {FREE_THRESHOLD_GB}GB")
    log("=" * 60)

    for name, url, note in GIT_REPOS:
        if not guard():
            break
        clone_git(name, url, note)

    for name, fid, note in GDOWN_FILES:
        if not guard():
            break
        gdrivedl(name, fid, note)

    for name, repo, note in HF_REPOS:
        if not guard():
            break
        hfdl(name, repo, note)

    log("=" * 60)
    log("全部队列执行完毕。最终磁盘剩余: %.1fGB" % free_gb())
    log("=" * 60)

if __name__ == "__main__":
    main()
