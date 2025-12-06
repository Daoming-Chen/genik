# GeNIK - 几何感知神经逆运动学

GeNIK (Geometry-aware Neural Inverse Kinematics) 是一个基于深度学习的逆运动学求解器，使用可微正运动学和三元组训练策略来学习机器人的逆运动学映射。

## 快速开始

### 1. 环境安装

```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 生成训练数据

为机器人生成训练数据集（包含FK采样、位姿聚类和一致性匹配）：

```bash
# 使用 Panda 机器人
python scripts/generate_dataset.py --urdf robots/panda_arm.urdf --output data/panda --num-triplets 5000000 --samples-per-pose 5

# 使用 UR10 机器人
python scripts/generate_dataset.py --urdf robots/ur10.urdf --output data/panda --num-triplets 5000000 --samples-per-pose 5
```

**主要参数说明：**
- `--urdf`: URDF 机器人模型文件路径
- `--output`: 数据输出目录
- `--num-samples`: FK 采样数量（默认 100000）
- `--num-clusters`: 位姿聚类数量（默认 1000）
- `--end-link`: 末端执行器链接名称（可选，默认使用最后一个链接）

生成的数据将保存在输出目录中：
- `train_triplets.pt`: 训练三元组数据
- `metadata.json`: 数据集元信息

### 3. 训练模型

使用生成的数据训练 GeNIK 模型：

```bash
# 训练 Panda 模型
python scripts/train_genik.py --urdf robots/panda_arm.urdf --data data/panda/train_triplets.pt --output outputs/panda --epochs 100

# 训练 UR10 模型
python scripts/train_genik.py --urdf robots/ur10.urdf --data data/ur10/train_triplets.pt --output outputs/ur10 --epochs 100
```

**主要参数说明：**
- `--urdf`: URDF 机器人模型文件路径
- `--data`: 训练数据文件路径
- `--output`: 模型输出目录（默认 `outputs/genik`）
- `--epochs`: 训练轮数（默认 100）
- `--batch-size`: 批次大小（默认 256）
- `--lr`: 学习率（默认 1e-3）
- `--resume`: 从检查点恢复训练（可选）

训练过程中会保存：
- `checkpoint.pt`: 最新检查点
- `best_model.pt`: 最佳模型（基于验证损失）
- TensorBoard 日志文件

**查看训练日志：**
```bash
tensorboard --logdir outputs/panda
```

### 4. 评估模型

评估训练好的模型性能：

```bash
# 评估 Panda 模型
python scripts/evaluate_genik.py --checkpoint outputs/panda/best_model.pt --data data/panda/train_triplets.pt --output results/panda

# 评估 UR10 模型
python scripts/evaluate_genik.py --checkpoint outputs/ur10/best_model.pt --data data/ur10/train_triplets.pt --output results/ur10
```

**主要参数说明：**
- `--checkpoint`: 模型检查点文件路径
- `--data`: 测试数据文件路径
- `--output`: 评估结果输出目录（可选）
- `--batch-size`: 批次大小（默认 256）
- `--visualize`: 是否生成可视化图表（默认 True）

评估结果包括：
- **Position Error**: 位置误差统计（均值、中位数、最大值）
- **Orientation Error**: 姿态误差统计
- **Success Rate**: 不同阈值下的成功率
- **Inference Time**: 推理速度
- 可视化图表（误差直方图、工作空间热图等）

## 完整流程示例

以下是完整的端到端验证流程：

```bash
# 1. 生成数据（约需 5-10 分钟）
python scripts/generate_dataset.py --urdf robots/panda_arm.urdf --output data/panda --num-samples 100000 --num-clusters 1000

# 2. 训练模型（约需 30-60 分钟，取决于硬件）
python scripts/train_genik.py --urdf robots/panda_arm.urdf --data data/panda/train_triplets.pt --output outputs/panda --epochs 100

# 3. 评估模型（约需 1-2 分钟）
python scripts/evaluate_genik.py --checkpoint outputs/panda/best_model.pt --data data/panda/train_triplets.pt --output results/panda
```

## 算法说明

GeNIK 使用以下关键技术：

1. **FK 采样**: 在关节空间均匀采样，生成 (q, pose) 对
2. **位姿聚类**: 使用 K-means 对位姿空间聚类，处理多解问题
3. **一致性匹配**: 为每个位姿找到一致性最好的关节配置三元组
4. **可微 FK**: 使用可微正运动学层实现端到端训练
5. **三元组损失**: 学习正确的多解分支选择

详细的算法描述请参考 `docs/algorithm.md` 和 `method.md`。

## 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_kinematics.py
pytest tests/test_network.py
```

## 依赖项

- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- scikit-learn >= 1.2.0
- 其他依赖见 `requirements.txt`

## 文档

- `docs/algorithm.md`: 算法详细说明
- `docs/api.md`: API 参考文档
- `docs/user_guide.md`: 用户指南
- `docs/examples.md`: 使用示例

## 许可证

[添加许可证信息]

## 引用

如果您使用 GeNIK，请引用：

```
[添加论文引用]
```
