# MDP 模块结构说明

`mdp/` 目录包含 Beyond Mimic 任务中 Markov Decision Process（马尔可夫决策过程）的各个组成部分。在 Isaac Lab 的 Manager-Based RL 框架中，MDP 被拆分为多个独立的子模块，每个模块负责 RL  pipeline 的一个方面。

---

## 目录结构

```
mdp/
├── __init__.py       # 模块入口，统一导出所有子模块
├── commands.py       # 命令（Command）定义 —— 告诉机器人"要做什么"
├── observations.py   # 观测（Observation）函数 —— 机器人"看到"的状态
├── rewards.py        # 奖励（Reward）函数 —— 驱动机器人学习的信号
├── events.py         # 事件（Event）函数 —— 域随机化与环境扰动
└── terminations.py   # 终止（Termination）条件 —— 判定 episode 何时结束
```

---

## 1. `__init__.py` — 模块入口

**作用**：作为 MDP 子包的入口，将所有子模块的函数和类统一导出，使得外部可以通过 `from mdp import ...` 直接访问所有 MDP 组件。

```python
from isaaclab.envs.mdp import *  # 继承 Isaac Lab 内置的 MDP 函数
from .events import *
from .observations import *
from .rewards import *
from .terminations import *
from .commands import *
```

**关键点**：先导入 Isaac Lab 内置的通用 MDP 函数，再导入本项目自定义的 MDP 函数（同名函数会覆盖内置版本）。

---

## 2. `commands.py` — 命令模块

**作用**：定义强化学习环境中的 **Command（命令/目标）**。在本任务中，命令来自于预录的运动捕捉（MoCap）数据，机器人需要跟踪这些运动轨迹。

### 核心类

#### `MotionLoader`
负责从 `.npz` 文件加载运动捕捉数据，包括：
- `joint_pos` / `joint_vel` — 关节位置和速度
- `body_pos_w` / `body_quat_w` — 身体在世界坐标系中的位置和姿态
- `body_lin_vel_w` / `body_ang_vel_w` — 身体的线速度和角速度
- `fps` — 运动数据的帧率
- `tail_len` — 轨迹尾部长度（用于避免采样到轨迹末尾）

#### `MotionCommand` (继承 `CommandTerm`)
核心命令类，负责在每个时间步更新运动跟踪目标：

| 方法 | 功能 |
|------|------|
| `_update_command()` | 每个 step 调用，推进时间步并计算相对位姿 |
| `_resample_command()` | 当运动播放完毕时，重新采样起始帧并重置机器人状态 |
| `_adaptive_sampling()` | 基于失败率的自适应采样策略，优先采样困难片段 |
| `_update_metrics()` | 计算跟踪误差指标（位置、姿态、速度等） |

**自适应采样机制**：将运动轨迹划分为多个 bin，记录每个 bin 的失败率。失败率高的 bin 有更高的采样概率，使得训练更集中于困难片段。

#### `MotionCommandCfg`
配置类，包含以下关键参数：

| 参数 | 类型 | 说明 |
|------|------|------|
| `asset_name` | `str` | 机器人资产名称 |
| `motion_file` | `str` | 运动数据文件路径 (.npz) |
| `anchor_body_name` | `str` | 锚点身体名称（用于计算相对位姿） |
| `body_names` | `list[str]` | 需要跟踪的身体部位列表 |
| `pose_range` | `dict` | 初始位姿随机范围 `{x, y, z, roll, pitch, yaw}` |
| `velocity_range` | `dict` | 初始速度随机范围 |
| `joint_position_range` | `tuple` | 关节位置随机范围 |
| `adaptive_*` | — | 自适应采样超参数 |

---

## 3. `observations.py` — 观测模块

**作用**：定义观测函数，将环境状态转换为神经网络的输入向量。

### 观测函数列表

| 函数 | 返回内容 | 维度 |
|------|----------|------|
| `robot_anchor_ori_w()` | 机器人锚点在世界坐标系中的朝向（旋转矩阵前两列展平） | `num_envs × 6` |
| `robot_anchor_lin_vel_w()` | 机器人锚点的线速度 | `num_envs × 3` |
| `robot_anchor_ang_vel_w()` | 机器人锚点的角速度 | `num_envs × 3` |
| `robot_body_pos_b()` | 各身体部位相对于锚点的位置（锚点坐标系） | `num_envs × (num_bodies*3)` |
| `robot_body_ori_b()` | 各身体部位相对于锚点的朝向（锚点坐标系） | `num_envs × (num_bodies*6)` |
| `motion_anchor_pos_b()` | 目标锚点相对于机器人锚点的位置 | `num_envs × 3` |
| `motion_anchor_ori_b()` | 目标锚点相对于机器人锚点的朝向 | `num_envs × 6` |

**设计思想**：观测值以 **机器人自身锚点为参考系** 来表示，这使得观测对全局位置不敏感，有助于泛化。

---

## 4. `rewards.py` — 奖励模块

**作用**：定义奖励函数，为强化学习提供训练信号。所有奖励函数返回一个标量张量。

### 奖励函数列表

| 函数 | 公式（核心） | 说明 |
|------|-------------|------|
| `motion_global_anchor_position_error_exp()` | `exp(-error / σ²)` | 惩罚锚点位置误差 |
| `motion_global_anchor_orientation_error_exp()` | `exp(-error / σ²)` | 惩罚锚点姿态误差 |
| `motion_relative_body_position_error_exp()` | `exp(-error / σ²)` | 惩罚身体部位相对位置误差 |
| `motion_relative_body_orientation_error_exp()` | `exp(-error / σ²)` | 惩罚身体部位相对姿态误差 |
| `motion_global_body_linear_velocity_error_exp()` | `exp(-error / σ²)` | 惩罚身体线速度误差 |
| `motion_global_body_angular_velocity_error_exp()` | `exp(-error / σ²)` | 惩罚身体角速度误差 |
| `feet_stance_time()` | 按站立时间奖励 | 鼓励足式机器人的步态周期 |

### 自适应 Sigma 机制

`_get_adaptive_sigma()` 函数维护每个奖励项的 EMA（指数移动平均）误差，并动态调整 σ 值：

```
σ = sqrt(min(EMA_error, historical_min_error))
```

这使得奖励尺度能够自动适应训练过程中的误差变化，保持奖励信号的稳定性。

---

## 5. `events.py` — 事件模块

**作用**：定义域随机化（Domain Randomization）函数，在 episode 开始时随机扰动环境参数以提高策略的鲁棒性。

### 事件函数列表

| 函数 | 功能 |
|------|------|
| `randomize_joint_default_pos()` | 随机化关节默认位置（模拟标定误差），支持 uniform / log_uniform / gaussian 分布 |
| `randomize_rigid_body_com()` | 随机化刚体质心（CoM）位置，模拟质量分布不确定性 |

**参数说明**：
- `env_ids` — 需要随机化的环境 ID 列表（`None` 表示全部）
- `asset_cfg` — `SceneEntityCfg` 配置，指定目标资产和关节/身体索引
- `operation` — 操作模式：`"abs"`（直接赋值）、`"add"`（叠加）、`"scale"`（缩放）

---

## 6. `terminations.py` — 终止条件模块

**作用**：定义 episode 的终止条件。当终止条件触发时，当前 episode 结束并重置。

### 终止条件函数列表

| 函数 | 触发条件 | 说明 |
|------|----------|------|
| `bad_anchor_pos()` | `‖anchor_pos_error‖ > threshold` | 锚点位置偏差过大 |
| `bad_anchor_pos_z_only()` | `‖anchor_pos_z_error‖ > threshold` | 锚点高度偏差过大 |
| `bad_anchor_ori()` | 重力投影偏差 > threshold | 锚点姿态偏差过大 |
| `bad_motion_body_pos()` | 任一身部位姿误差 > threshold | 身体部位位置偏差过大 |
| `bad_motion_body_pos_z_only()` | 任一身部位姿 z 误差 > threshold | 身体部位高度偏差过大 |

**设计思想**：终止条件以 anchor（锚点）为参考系来判断，仅当误差超过阈值时才终止 episode。这允许策略在训练早期有一定的探索容忍度。

---

## 整体数据流

```
┌─────────────────────────────────────────────────────────────┐
│                     RL Episode Loop                          │
├─────────────────────────────────────────────────────────────┤
│  1. Events (events.py)                                       │
│     └─ 随机化关节零位、质心等物理参数                          │
│                         ↓                                    │
│  2. Commands (commands.py)                                   │
│     └─ 从 MoCap 数据采样目标运动，计算相对位姿                  │
│                         ↓                                    │
│  3. Observations (observations.py)                           │
│     └─ 从环境 & Command 提取观测向量 → 输入 Policy            │
│                         ↓                                    │
│  4. Policy 推理 → Action → 执行仿真步                         │
│                         ↓                                    │
│  5. Rewards (rewards.py)                                     │
│     └─ 计算跟踪误差奖励                                      │
│                         ↓                                    │
│  6. Terminations (terminations.py)                           │
│     └─ 判断是否因误差过大而终止                                │
│                         ↓                                    │
│  7. Commands._update_command()                               │
│     └─ 推进运动时间步，如播放完毕则自适应重采样                 │
└─────────────────────────────────────────────────────────────┘
```

## 与 Isaac Lab 框架的关系

本 `mdp/` 模块遵循 Isaac Lab 的 `ManagerBasedRLEnv` 架构，通过统一的 Manager 接口注册各组件：

- **CommandManager** — 管理 `commands.py` 中定义的命令
- **ObservationManager** — 管理 `observations.py` 中定义的观测函数
- **RewardManager** — 管理 `rewards.py` 中定义的奖励函数
- **EventManager** — 管理 `events.py` 中定义的事件函数
- **TerminationManager** — 管理 `terminations.py` 中定义的终止条件

每个 Manager 通过配置文件（如 `env_cfg.py`）中的字典进行配置，将函数名与参数绑定。
