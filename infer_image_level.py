# -*- coding: utf-8 -*-
"""
infer_image_level.py —— 竞赛提交推理：图像级火焰 0/1。

用法:
  python infer_image_level.py --model <best.pt> --source <图片目录或单图> --conf 0.35 --out result.json

规则(与竞赛一致):
- 仅"火焰(类0)"触发"有火"；烟雾(类1)不触发，以免影响 Precision。
- 一张图内只要有任一 类0 框 conf>=阈值 -> 该图=1，否则=0。
- 输出 json: {"图像名(不含扩展名)": 0/1, ...}
"""
import argparse, json, os, glob
from pathlib import Path
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="训练好的 yolo26*.pt")
    ap.add_argument("--source", required=True, help="测试图片目录或单张图片")
    ap.add_argument("--conf", type=float, default=0.35, help="火焰判定阈值(图像级)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="result.json")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    model = YOLO(args.model)
    # 收集图片
    if os.path.isdir(args.source):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        imgs = []
        for e in exts:
            imgs += glob.glob(os.path.join(args.source, e))
    else:
        imgs = [args.source]

    results = model.predict(
        source=imgs, imgsz=args.imgsz, conf=args.conf, iou=0.45,
        device=args.device, workers=8, batch=16, stream=True,
        verbose=False,
    )

    out = {}
    for r in results:
        name = os.path.splitext(os.path.basename(r.path))[0]
        boxes = r.boxes
        fire = False
        if boxes is not None and len(boxes) > 0:
            cls = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()
            # 仅火焰(类0)且达到阈值
            fire = any(c == 0 and cf >= args.conf for c, cf in zip(cls, conf))
        out[name] = 1 if fire else 0

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    n1 = sum(1 for v in out.values() if v == 1)
    print(f"[infer] 共 {len(out)} 张, 判为有火 {n1} 张, 无火 {len(out)-n1} 张")
    print(f"[infer] 已写出 -> {args.out}")


if __name__ == "__main__":
    main()
