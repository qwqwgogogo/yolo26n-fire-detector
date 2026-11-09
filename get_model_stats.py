"""读取蒸馏产线模型, 输出真实参数量/FLOPs/层数, 供与论文对标.
只读, 不训练.
"""
import os, sys, json

ENV = "C:/Users/qwqw/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
# 该脚本由外部用上述解释器运行; 此处仅逻辑
MODEL = r"E:\火情监测\火情监测\distill_pipeline\final_yolo26n_best.pt"

def main():
    from ultralytics import YOLO
    m = YOLO(MODEL)
    # model.info 返回 (layers, params, grads, flops)
    info = m.info()
    # info 可能是 tuple 或 Model; 兼容处理
    try:
        layers, params, grads, flops = info
    except Exception:
        # 某些版本 info() 打印并返回 None; 用 model.model 统计
        md = m.model
        params = sum(p.numel() for p in md.parameters())
        flops = None
        layers = None
    out = {
        "model": MODEL,
        "params_M": round(params / 1e6, 3) if params else None,
        "params": int(params) if params else None,
        "flops_G": round(flops / 1e9, 3) if flops else None,
        "layers": layers,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
