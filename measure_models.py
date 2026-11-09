import os
from ultralytics import YOLO

base = r"E:/火情监测/火情监测"
paths = {
    "teacher_YOLOv11s": os.path.join(base, "external_datasets/teachers/leeyunjai_yolo11_firedetect/firedetect-11s.pt"),
    "teacher_YOLOv11x": os.path.join(base, "external_datasets/teachers/leeyunjai_yolo11_firedetect/firedetect-11x.pt"),
    "student_YOLOv26n": os.path.join(base, "distill_pipeline/final_yolo26n_best.pt"),
}

print(f"{'model':18s} {'file_MB':>8s} {'params':>12s} {'GFLOPs':>8s}")
print("-" * 52)
for name, p in paths.items():
    if not os.path.exists(p):
        print(f"{name:18s} MISSING: {p}")
        continue
    size_mb = os.path.getsize(p) / 1024 / 1024
    params = 0
    gflops = float("nan")
    try:
        m = YOLO(p)
        params = sum(x.numel() for x in m.model.parameters())
        info = m.model.info(verbose=False)
        try:
            gflops = float(getattr(info, "GFLOPs", 0) or 0)
        except Exception:
            gflops = float("nan")
    except Exception as e:
        print(f"{name:18s} LOAD_ERR {e}")
        continue
    print(f"{name:18s} {size_mb:8.1f} {params:12,} {gflops:8.1f}")
