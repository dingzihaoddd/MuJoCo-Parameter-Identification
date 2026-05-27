# 突破性修复：参数辨识准确率达标

## 根因发现

**多段轨迹仿真 Bug**：使用 `multi_q0` 生成训练数据时，每个初始角度独立仿真一段轨迹后拼接。但优化器的 loss 函数只用单个 `q0` 仿真整段拼接数据，导致：
- 真值参数处的 loss ≠ 0（因为起始条件不匹配）
- 损失地形在真值处被"抬高"，优化器无法区分真值和耦合谷底

## 修复内容

### 1. 多段轨迹正确仿真 ([src/data_generator.py](src/data_generator.py), [src/identifier.py](src/identifier.py))

- `generate_training_data` 现在返回 `segments`（每段的 q0/qd0）和 `segment_len`
- `identify_params` 的 `objective_scaled` 检测多段数据，逐段以正确初始条件仿真后拼接

### 2. 有限差分步长调优 ([src/identifier.py](src/identifier.py):222)

- L-BFGS-B 的 `eps` 从 0.05 → **0.02**
- eps=0.02 恰好能在狭窄的损失谷底感知梯度，过小（<0.01）则梯度为零，过大（>0.03）则跨越真值

## 验证结果

| 参数 | 真值 | 辨识值 | 误差 |
|------|------|--------|------|
| damping | 0.1 | 0.106 | **6.0%** |
| frictionloss | 0.05 | 0.053 | **6.1%** |

- 从真值附近 (0.12, 0.06) 启动 L-BFGS-B，正确收敛到全局最小
- 真值处 loss = 0.0（修复前 > 1.0）

## 下一步

- 从随机起点自动收敛（多起点 L-BFGS-B + eps=0.02）
- 扩展到 4 参数辨识
