# -*- coding: utf-8 -*-
import os, glob, sys
from collections import Counter

def log(*a):
    print(*a, flush=True)

def scan(name, label_root):
    cls = Counter()
    n_files = 0
    empty = 0
    fire = 0
    smoke_only = 0
    fire_and_smoke = 0
    other = 0
    for lp in glob.glob(os.path.join(label_root, "**", "*.txt"), recursive=True):
        n_files += 1
        try:
            with open(lp, encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            continue
        lines = [l for l in content.splitlines() if l.strip()]
        if not lines:
            empty += 1
            continue
        cset = set()
        for l in lines:
            p = l.split()
            if len(p) < 5:
                continue
            try:
                c = int(p[0])
            except Exception:
                continue
            cls[c] += 1
            cset.add(c)
        if 0 in cset and 1 in cset:
            fire_and_smoke += 1
        elif 0 in cset:
            fire += 1
        elif 1 in cset:
            smoke_only += 1
        else:
            other += 1
    log("### %s (%s)" % (name, label_root))
    log("  label文件数=%d" % n_files)
    log("  类别分布(class:count)= %s" % dict(sorted(cls.items())))
    log("  纯火图=%d 纯烟图=%d 火+烟=%d 空背景=%d 其他类=%d" % (fire, smoke_only, fire_and_smoke, empty, other))
    log("  >>> 阴性候选(无火框:纯烟+空背景+其他)= %d" % (smoke_only + empty + other))

base = "E:/火情监测/火情监测/external_datasets"
scan("D-Fire", os.path.join(base, "academic_D-Fire"))
scan("hiennguyen", os.path.join(base, "HF_hiennguyen_FireSmoke", "labels", "train"))
log("DONE")
