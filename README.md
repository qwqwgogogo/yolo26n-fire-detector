# 火情监测 🔥

> 基于 **YOLOv26n** 训练的监控场景火情检测模型，通过 **教师—学生跨架构蒸馏 + 伪标签合成** 得到，可在单张监控图上做 **图像级"有/无火"判定**，也可输出火焰/烟雾目标框。

本仓库以 **开源模型权重 + 复现所需的训练配方与代码** 为核心；原始训练数据集与伪标签因体积和版权原因 **不随仓库提供**（数据准备脚本已包含，自行准备即可复现）。

## 模型文件

| 文件 | 说明 | 大小 |
|---|---|---|
| **`final_yolo26n_best.pt`** | **最终模型**（推荐提交/使用模型，YOLOv26n 微调后的 best 权重） | 5.4 MB |
| `final_expA_best.pt` / `final_expB_best.pt` / `final_exp2_best.pt` / `final_ext_yolo26n_best.pt` | 其他蒸馏实验的 best 权重（同规格，可选） | 各 ~5.4 MB |
| `yolo26n.pt` / `yolo26s.pt` / `yolo26m.pt` | 官方预训练基础权重（底座，可官方仓库重新下载） | 5.4 / 20.4 / 43.2 MB |

## 这个模型是怎么来的（训练配方）

### 1. 数据
- **官方竞赛集**：1100 张（859 有火 / 241 无火），按 880 训 / 220 验划分，另 63 张无标注图作为负样本参与训练以降低误检。
- **外部公开数据集**：`DBA-Fire`、`FireSmoke-Txinyan` 等（约 1 万+ 张火情/烟雾图）。
- **合成与增强**：数据池上做了增强镜像、假火负样本、难例挖掘、教师伪标签，组成蒸馏训练集。

> 数据准备脚本：`build_external_train.py`、`build_neg_dataset.py`、`gen_pseudo.py`、`mine_targeted_pool.py`、`split_dataset.py`。复现时请自行下载原数据集（各自来源 license 见下文），运行这些脚本生成训练集。

### 2. 教师—学生蒸馏（核心思路）
| 角色 | 模型 | 说明 |
|---|---|---|
| **教师** | **YOLOv11s**（`firedetect-11s.pt`） | 在公开数据上训练，输出框级伪标签；类别 `Fire=0 / Smoke=1` |
| **学生** | **YOLOv26n**（2.5M 参数） | 用教师伪标签 + 增强数据做标准 YOLOv26 训练（**跨架构蒸馏**） |

蒸馏方式为 **离线伪标签合并 + YOLOv26 标准训练**（L0 路线）：教师先在公开数据上产出伪标签，再合并进训练池让学生训练。

### 3. 训练配置（`config_expA.json`）
```json
{
  "base_model": "yolo26n.pt",
  "model_size": "n",
  "epochs": 40,
  "patience": 15,
  "imgsz": 640,
  "enhance": false
}
```

## 指标（验证集，仅评火焰类 0 的图像级 P/R）

| 模型 | 最佳 conf | Precision | Recall | F1 |
|---|---|---|---|---|
| **yolo26n**（最终模型） | **0.10** | **0.9318** | **0.9011** | **0.9162** |

## 如何使用（图像级判断）

核心逻辑：**对一张图，若存在任何 `conf ≥ 阈值` 的火焰框（类 0），则判定该图"有火"；烟雾（类 1）不触发"有火"判定（避免伤 Precision）。**

```python
from ultralytics import YOLO

model = YOLO("final_yolo26n_best.pt")

def image_has_fire(image_path, conf=0.10, fire_class=0):
    res = model.predict(image_path, conf=conf, verbose=False)
    for r in res:
        for box in r.boxes:
            if int(box.cls) == fire_class and float(box.conf) >= conf:
                return 1  # 有火
    return 0             # 无火
```

**阈值建议**：验证最佳 `conf=0.10`；竞赛提交用 `conf=0.05`（更看重召回/少漏火）。对 1100 张图运行输出 `{图名: 0/1}` 即得结果。

## 项目结构（开源版）

```
├── final_yolo26n_best.pt          # 最终模型
├── final_*.pt                     # 其他实验权重
├── yolo26n.pt / yolo26s.pt / yolo26m.pt   # 基础权重
├── README.md                      # 本文件（模型由来与配方）
├── RESULT.md                      # 蒸馏训练结果汇总（含指标）
├── config_expA.json / experiments_plan.json / status.json
├── train.py / preprocess.py / loadvideodata.py   # 训练 / 推理脚本
├── optimize_loop.py / fire_launcher.py / run_distill.py / gen_pseudo.py / exp_train_eval.py
│   infer_image_level.py / teacher_eval_full.py 等  # 蒸馏/评估/数据准备脚本
└── .md 文档                        # 选型/方案/数据集清单等背景文档
```

## 环境依赖

- Python 3.8+
- `pip install ultralytics opencv-python torch`
- 推理：`python infer_image_level.py --conf 0.05 <测试集目录>`（输出 `result.json`）
- 图片推理：`python preprocess.py [path/to/image.jpg]`
- 视频流：`python loadvideodata.py [video.mp4 | 0]`，按 `Q` 退出

## 许可证（License）

- **基础权重 YOLOv26**：源自 [ultralytics/YOLO](https://github.com/ultralytics/ultralytics)，**AGPL-3.0**，见其 `LICENSE`。
- **本仓库代码**：建议 **MIT**（请在你真正开源时确认/替换）。
- **训练所用的外部数据集**（DBA-Fire、FireSmoke 等）各自保留原版权，本仓库**不重分发的原始图片/标签/测试集**。
- **模型权重**：请在你开源时明确声明该权重的使用/分发条款（AGPL 的传染性取决于你如何分发源码与权重，建议在仓库顶部 `LICENSE` 与模型说明中写清）。

## 复现

1. 准备数据：下载官方集 + 外部公开集 → 运行 `split_dataset.py`、`build_*`、`gen_pseudo.py` 等生成训练集（脚本在 `distill_pipeline/` 原始目录；开源后脚本已并入仓库根）。
2. 训练：`python -c "from ultralytics import YOLO; YOLO('yolo26n.pt').train(data='train.yml', epochs=40, patience=15, imgsz=640)"`（按 `config_expA.json`）。
3. 取 `best` 权重即得 `final_yolo26n_best.pt`。

## 🤖 自动部署说明

> 本仓库的全部上线流程（本地清理数据 → git 提交 → 仓库创建 → push 发布）均由 **DeepSeek Harness (DSH)** 自动完成，运行的语言模型为 **DeepSeek `bonsai2-27b`**。
>
> 本次 push 上线的初衷并非真实生产部署，而是**测试该模型的性能是否足以独立、无人值守地完成一整套「数据清理 + git 提交 + 仓库创建 + push 上线」的部署动作**。
