# 蒸馏训练结果汇总

- 生成时间: 2026-08-09 14:00:15
- 教师: YOLOv11 (firedetect-11x/11s)  Fire=0, Smoke=1
- 蒸馏方式: 离线伪标签合并 + YOLOv26 标准训练(跨架构蒸馏)

## 各尺寸图像级火焰 P/R（验证集，仅评火焰类0）

| 尺寸 | 最佳conf | Precision | Recall | F1 | 权重 |
|---|---|---|---|---|---|
| yolo26n | 0.10 | 0.9318 | 0.9011 | 0.9162 | E:\火情监测\火情监测\distill_pipeline\final_yolo26n_best.pt |

## 推荐提交模型: yolo26n
- 权重: E:\火情监测\火情监测\distill_pipeline\final_yolo26n_best.pt
- 图像级火焰决策阈值 conf=0.10（类0框 conf>=此值 -> 图=1）
- 验证集 P=0.9318 R=0.9011 F1=0.9162

## 下一步（竞赛提交）
- 用 `infer_image_level.py` 对官方测试集推理，输出 result.json（image_name -> 0/1）。
- 仅火焰(类0)触发'有火'；烟雾(类1)不得触发，以免影响 Precision。
- 测试集提前3天公布，下载后放入指定目录再推理即可。