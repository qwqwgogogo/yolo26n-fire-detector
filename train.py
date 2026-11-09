import os
from ultralytics import YOLO, settings


def main():
    # 将 YOLO 输出目录固定到本项目下的 runs/（覆盖全局指向 Stable Diffusion 的路径）
    settings.update({'runs_dir': os.path.join(os.getcwd(), 'runs')})

    # 加载预训练权重作为基础模型
    model = YOLO('yolo26n.pt')

    # === 本机硬件配置 ===
    #   GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU (12GB, Blackwell, compute capability 12.0)
    #   CPU: 32 逻辑核心
    #   RAM: 32GB
    # 目标：最大化利用 GPU 与多核 CPU，尽快完成训练。
    model.train(
        data="dataset_root/dataset.yml",
        epochs=100,
        imgsz=640,
        batch=-1,           # -1 = 自动探测显存可承受的最大 batch，榨干 GPU 显存
        device=0,            # 使用 GPU 0 (RTX 5070 Ti Laptop GPU)
        workers=0,           # Windows spawn 下 cache='ram' 会把 0.6GB 图片随 dataset pkl 给每个子进程，
                              # 多 worker 同时解 pkl 直接 MemoryError；设为 0 在主进程喂数据，
                              # 图片已常驻 RAM 缓存，GPU 仍是瓶颈，速度基本无损。
        cache='ram',         # 32GB 内存充足，把图片缓存到内存，消除磁盘 IO 瓶颈
        amp=True,            # 混合精度训练，GPU 上大幅提速且基本不损精度
        optimizer='auto',    # 自动选用 AdamW
        cos_lr=True,         # 余弦退火学习率，收敛更稳
        close_mosaic=10,     # 最后 10 轮关闭 mosaic 增强，提升最终 mAP
        patience=50,         # 验证指标 50 轮无提升则早停，避免浪费时间
        name="yolo26n_gpu",
        exist_ok=True,       # 覆盖上一轮训练结果
    )


if __name__ == '__main__':
    # Windows 下 multiprocessing 使用 spawn 模式，子进程会 re-import 本模块。
    # 必须用此保护，否则子进程启动时会重复执行 model.train() 而崩溃。
    main()
