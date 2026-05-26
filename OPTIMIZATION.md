# MuJoCo 动力学参数辨识 — 优化策略总结

## 问题背景

通过最小化仿真轨迹与真实轨迹的均方误差（MSE），从数据中辨识单关节摆锤的动力学参数（mass, damping, frictionloss, armature）。

## 遇到的问题与解决

### 1. 中文字体乱码

matplotlib 默认字体不支持中文。

**修复**: 设置 `rcParams["font.sans-serif"] = ["SimHei", ...]`，同时 `axes.unicode_minus = False` 避免负号异常。

### 2. L-BFGS-B 2 次迭代即收敛

**现象**: 优化器输出 `CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH`，仅 2-3 次迭代就停止，参数几乎不更新。

**根因**: 参数尺度差异大（mass ~1.0, armature ~0.01），默认有限差分步长对各参数不均，梯度计算失准。

**修复**:
- 参数内部缩放到 [0, 1]，使各维度有限差分精度一致
- 设置 `ftol=1e-15`, `gtol=1e-8` 防止过早收敛
- 增大有限差分步长 `eps=0.05`

### 3. 损失地形"平地+针尖"

**现象**: 损失在真值处为 0，但偏离真值后迅速跳到 ~80 且几乎不变（用 multisine 激励）。梯度方向不可靠，优化器迷失。

**根因**: multisine 激励产生的轨迹对参数变化不够敏感，损失地形在远离真值处极其平坦。

**修复**: 改用单频正弦激励（0.7 Hz，接近共振频率），使响应幅度对阻尼/摩擦参数高度敏感。损失地形变得平滑且呈凸形。

### 4. 阻尼与摩擦的耦合

**现象**: 即使损失地形平滑，优化器仍收敛到错误值（如 damping=0.04, frictionloss=0.09，真值为 0.1 和 0.05）。

**根因**: damping 和 frictionloss 对轨迹的影响相似（都抑制运动），存在一个"耦合谷底"——不同 (d, f) 组合产生几乎相同的损失值。真值处的全局最小值非常窄，优化器容易落入较宽的局部最小。

**修复**:
- 使用多个初始角度（10°, 30°, 60°, -20°, -45°）生成训练数据并拼接。不同初始角度产生不同速度剖面，帮助区分速度比例阻尼与符号函数摩擦。
- 改用差分进化（differential_evolution）做全局搜索，避免陷入局部最优。
- 缩小参数搜索范围（damping: [0.001, 1.0], frictionloss: [0.001, 0.5]），提高搜索密度。

## 关键改动文件

| 文件 | 改动 |
|------|------|
| [src/identifier.py](src/identifier.py) | 参数缩放、支持自定义参数子集和边界、三种优化方法（L-BFGS-B / Nelder-Mead / differential_evolution） |
| [src/data_generator.py](src/data_generator.py) | 新增 `generate_sine`、`generate_multisine`，支持 `multi_q0` 多初始角度拼接 |
| [src/simulator.py](src/simulator.py) | 构造函数支持 `timestep` 参数覆盖，加速仿真 |
| [main.py](main.py) | 多次随机重启、结果保存到 `results/<时间戳>/` 目录 |

## 使用方式

```bash
python main.py
```

结果保存在 `results/YYYYMMDD_HHMMSS/` 目录下，包含：
- `results.png` — 轨迹对比、损失曲线、参数柱状图
- `results.txt` — 数值结果文本
