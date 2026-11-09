from ultralytics import YOLO

# 加载预训练 YOLO 模型（用于验证模型可正常加载与推理）
model = YOLO('yolo26n.pt')

# 对示例图片进行推理，展示并保存结果
url = 'https://ultralytics.com/images/zidane.jpg'
model(url, show=True, save=True)
