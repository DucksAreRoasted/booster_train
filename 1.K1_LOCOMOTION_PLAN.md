# K1 纯本体 Locomotion 训练方案

> 目标：训练 K1 机器人在平地、坡面、小扰动、低台阶地形上稳定前进、侧移、转向。不加入相机。

---

## 目录

1. [架构概述](#1-架构概述)
2. [文件清单与变更](#2-文件清单与变更)
3. [命令系统设计](#3-命令系统设计)
4. [观测空间设计](#4-观测空间设计)
5. [奖励函数设计](#5-奖励函数设计)
6. [终止条件设计](#6-终止条件设计)
7. [地形配置](#7-地形配置)
8. [域随机化与扰动](#8-域随机化与扰动)
9. [课程学习策略](#9-课程学习策略)
10. [机器人配置](#10-机器人配置)
11. [PPO 超参数](#11-ppo-超参数)
12. [训练与评估命令](#12-训练与评估命令)
13. [实现步骤](#13-实现步骤)
14. [预期里程碑](#14-预期里程碑)

---

## 1. 架构概述

### 1.1 核心变化：从 Motion Tracking → Velocity Tracking

| 维度 | 当前 (Beyond Mimic) | 目标 (Locomotion) |
|------|---------------------|-------------------|
| 命令类型 | `MotionCommand` — 逐帧关节/身体目标 | `UniformVelocityCommand` — 速度指令 (vx, vy, ωz) |
| 策略目标 | 复制 MoCap 参考动作 | 跟踪速度指令同时保持平衡 |
| 奖励函数 | 身体部位位姿/速度匹配 MoCap | 速度跟踪 + 生存 + 能耗 + 姿态正则化 |
| 观测空间 | 锚点相对位姿 + 目标关节角度 | 速度指令 + 关节状态 + 基座状态 |
| 控制关节 | 全部 22 DOF（含头、臂） | 仅下肢 12 DOF（锁定上半身） |
| 地形 | 近乎平坦（波动 <1.5cm） | 平地 + 坡面 + 台阶 + 离散障碍 + 波浪 |

### 1.2 数据流

```
UniformVelocityCommand          Policy (PPO Actor)          仿真步进
┌─────────────────────┐        ┌─────────────────┐        ┌──────────┐
│ vx: 0.5 m/s         │        │                 │        │          │
│ vy: 0.0 m/s    ──────┼──► Obs │ MLP             │──► Act │ 12 joint │──► next obs
│ ωz: 0.0 rad/s       │        │ [512,256,128]   │  pos   │ targets  │
│ standing: False      │        │                 │        │          │
└─────────────────────┘        └─────────────────┘        └──────────┘
        │                                                       │
        └───────────────── resample every ──────────────────────┘
                          5-10 seconds
```

---

## 2. 文件清单与变更

### 2.1 新建文件

```
source/booster_train/booster_train/tasks/manager_based/beyond_mimic/
├── robots/k1/locomotion/
│   ├── __init__.py               # Gym 注册
│   ├── env_cfg.py                # 环境配置（Flat / Rough 变体）
│   ├── tracking_env_cfg.py       # MDP 配置（观测/奖励/终止/事件）
│   └── ppo_cfg.py                # PPO 超参
└── mdp/
    ├── locomotion_commands.py    # VelocityCommand 封装（可选，如需要定制）
    ├── locomotion_rewards.py     # Locomotion 专用奖励函数
    └── locomotion_observations.py # Locomotion 专用观测函数（可选）
```

### 2.2 修改文件

| 文件 | 变更 |
|------|------|
| `booster_assets/src/booster_assets/__init__.py` 或新建 robot cfg | 新增 `BOOSTER_K1_LOCOMOTION_CFG`（仅下肢关节的 ArticulationCfg） |
| `source/.../mdp/__init__.py` | 添加新模块的 import（如需新 mdp 文件） |

### 2.3 不修改的文件

- `commands.py`、`rewards.py`、`observations.py`、`terminations.py`、`events.py` — 现有 Beyond Mimic 管线保持不动
- 所有 `fight_001/`、`indian_dance/`、`mj_dance_*` 目录 — 保持不动

---

## 3. 命令系统设计

使用 Isaac Lab 内置的 `UniformVelocityCommand`（来自 `isaaclab.envs.mdp.commands.velocity_command`）。

### 3.1 速度指令范围

针对 K1 的体型（身高约 1.2m，Trunk 质量 6.5kg），初始阶段使用保守的速度范围：

```python
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg

UniformVelocityCommandCfg(
    asset_name="robot",
    heading_command=False,          # 直接使用角速度指令，不转换为朝向
    rel_standing_envs=0.1,          # 10% 的环境发出站立指令（速度为 0）
    resampling_time_range=(5.0, 10.0),  # 每 5-10 秒换一次指令
    ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.5, 0.8),      # 前进速度：-0.5~0.8 m/s（前进略快于后退）
        lin_vel_y=(-0.4, 0.4),      # 侧移速度：±0.4 m/s
        ang_vel_z=(-1.0, 1.0),      # 转向角速度：±1.0 rad/s
    ),
)
```

### 3.2 设计决策

| 参数 | 选择 | 理由 |
|------|------|------|
| `heading_command` | `False` | 简化——直接采样角速度，不做 heading 跟踪。后续阶段可改为 `True` |
| `rel_standing_envs` | `0.1` | 10% 时间静止，策略需学会站立不动 |
| 前进速度上限 | 0.8 m/s | K1 为小体型人形机器人，0.8 m/s 约相当于人类快步走 |
| `resampling_time_range` | (5, 10) 秒 | 比默认的 (10, 10) 更有变化，策略在同一个指令上需要坚持足够长时间 |

---

## 4. 观测空间设计

### 4.1 观测项

```python
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 1. 速度指令 (3 维)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "velocity"})

        # 2. 基座线速度 (3 维)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))

        # 3. 基座角速度 (3 维)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))

        # 4. 投影重力方向 (3 维) — 隐式提供基座姿态
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.02, n_max=0.02))

        # 5. 关节位置（相对默认值）(12 维，仅下肢)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                ".*_Hip_Pitch", ".*_Hip_Roll", ".*_Hip_Yaw",
                ".*_Knee_Pitch", ".*_Ankle_Pitch", ".*_Ankle_Roll",
            ])
        }, noise=Unoise(n_min=-0.01, n_max=0.01))

        # 6. 关节速度 (12 维)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                ".*_Hip_Pitch", ".*_Hip_Roll", ".*_Hip_Yaw",
                ".*_Knee_Pitch", ".*_Ankle_Pitch", ".*_Ankle_Roll",
            ])
        }, noise=Unoise(n_min=-0.1, n_max=0.1))

        # 7. 上一帧动作 (12 维)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
```

### 4.2 观测维度汇总

| 观测项 | 维度 | 说明 |
|--------|------|------|
| velocity_commands | 3 | vx, vy, ωz |
| base_lin_vel | 3 | 基座线速度（本体坐标系）|
| base_ang_vel | 3 | 基座角速度（本体坐标系）|
| projected_gravity | 3 | 重力在基座坐标系下的投影 |
| joint_pos | 12 | 下肢关节位置 - 默认值 |
| joint_vel | 12 | 下肢关节速度 |
| actions | 12 | 上一帧输出的关节目标 |
| **总计（平地）** | **48** | |
| height_scan（粗糙地形） | 49 (7×7 grid) | 躯干下方地形高度采样，offset=0.0 |
| **总计（粗糙地形）** | **97** | |

### 4.3 场景配置补充

粗糙地形需要 RayCaster 做 height scan：

```python
from isaaclab.sensors import RayCasterCfg, patterns

class MySceneCfg(InteractiveSceneCfg):
    # ... terrain, robot, lights, contact_forces ...
    
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.6, 0.6]),  # 7×7=49 rays
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
```

### 4.4 Privileged（Critic）观测

Privileged 观测可用于 teacher-student 蒸馏。第一版不引入非对称结构，policy 和 critic 使用相同观测。

---

## 5. 奖励函数设计

### 5.1 奖励项总览

| # | 奖励项 | 权重 | 类型 | 说明 |
|---|--------|------|------|------|
| 1 | `track_lin_vel_xy_yaw_frame_exp` | **2.0** | 正 | Yaw-aligned XY 线速度跟踪（指数核 `exp(-error²/σ²)`，σ=0.25） |
| 2 | `track_ang_vel_z_world_exp` | **1.0** | 正 | World-frame 偏航角速度跟踪（指数核，σ=0.25） |
| 3 | `lin_vel_z_l2` | **-2.0** | 负 | 惩罚 Z 轴线速度（阻止跳跃/下沉） |
| 4 | `ang_vel_xy_l2` | **-0.5** | 负 | 惩罚 roll/pitch 角速度（保持姿态稳定） |
| 5 | `flat_orientation_l2` | **-1.0** | 负 | 惩罚基座倾斜（投影重力的 xy 分量） |
| 6 | `base_height_l2` | **-1.0** | 负 | 惩罚基座偏离目标高度 |
| 7 | `joint_torques_l2` | **-1e-4** | 负 | 惩罚关节力矩（仅下肢，能耗最小化） |
| 8 | `joint_vel_l2` | **-1e-3** | 负 | 惩罚关节速度（平滑动作） |
| 9 | `joint_acc_l2` | **-2.5e-7** | 负 | 惩罚关节加速度（避免抖动） |
| 10 | `action_rate_l2` | **-0.5** | 负 | 惩罚动作变化率（动作平滑） |
| 11 | `joint_pos_limits` | **-5.0** | 负 | 惩罚关节限位违反 |
| 12 | `undesired_contacts` | **-5.0** | 负 | 惩罚非脚底的碰撞（膝盖、躯干触地） |
| 13 | `feet_air_time_positive_biped` | **0.5** | 正 | 双足交替离地奖励（threshold=0.4） |
| 14 | `feet_slide` | **-0.25** | 负 | 惩罚脚底滑动 |
| 15 | `joint_deviation_l1` | **-0.1** | 负 | 惩罚髋关节偏离默认姿态（仅 Hip_Yaw + Hip_Roll） |
| 16 | `is_terminated` | **-200.0** | 负 | 摔倒终止惩罚 |

### 5.2 关键奖励实现细节

#### 5.2.1 速度跟踪

Isaac Lab 有两套速度跟踪函数。**推荐使用 yaw-aligned 版本**——H1/G1 人形机器人的标准做法：

```python
# 方案 A（推荐）：yaw-aligned 坐标系 + world 角速度
# 来自 isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards
track_lin_vel_xy_yaw_frame_exp = RewTerm(
    func=mdp_locomotion.track_lin_vel_xy_yaw_frame_exp,
    weight=2.0,
    params={"command_name": "velocity", "std": 0.25},
)
track_ang_vel_z_world_exp = RewTerm(
    func=mdp_locomotion.track_ang_vel_z_world_exp,
    weight=1.0,
    params={"command_name": "velocity", "std": 0.25},
)

# 方案 B（备选）：base-frame 版本（内置 isaaclab.envs.mdp）
# 适合初期快速验证
track_lin_vel_xy_exp = RewTerm(
    func=mdp.track_lin_vel_xy_exp,
    weight=2.0,
    params={"command_name": "velocity", "std": 0.25},
)
track_ang_vel_z_exp = RewTerm(
    func=mdp.track_ang_vel_z_exp,
    weight=1.0,
    params={"command_name": "velocity", "std": 0.25},
)
```

> **yaw-aligned 版本的优势**：人形机器人在行走时躯干会有自然的 roll/pitch 摆动。Base-frame 速度跟踪会被这种摆动干扰，而 yaw-aligned frame 只保留偏航旋转，消除了躯干摆动对速度跟踪的噪声。

#### 5.2.2 Feet Air Time（biped 专用版本）

Isaac Lab Tasks 提供了 `feet_air_time_positive_biped`（位于 `isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards`），专为双足设计：

```python
feet_air_time = RewTerm(
    func=mdp_locomotion.feet_air_time_positive_biped,
    weight=0.5,
    params={
        "command_name": "velocity",
        "sensor_cfg": SceneEntityCfg(
            "contact_forces",
            body_names=["left_foot_link", "right_foot_link"],
        ),
        "threshold": 0.4,
    },
)
```

该函数的智能之处：只在速度指令 > 0.1 m/s 且处于单支撑相（一只脚离地）时给奖励——完美契合双足行走的交替步态。

#### 5.2.3 Feet Slide（自定义，参考 H1/G1）

惩罚脚在接触地面时的滑动：

```python
# 来自 isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards
feet_slide = RewTerm(
    func=mdp_locomotion.feet_slide,
    weight=-0.25,
    params={
        "sensor_cfg": SceneEntityCfg(
            "contact_forces",
            body_names=["left_foot_link", "right_foot_link"],
        ),
        "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
    },
)
```

#### 5.2.4 Height Scan 观测（粗糙地形需要）

当使用粗糙地形时，加入 height_scan 观测让策略感知脚下地形：

```python
# 需要先在 scene 中添加 height_scanner (RayCaster)
height_scan = ObsTerm(
    func=mdp.height_scan,
    params={"sensor_cfg": SceneEntityCfg("height_scanner")},
    noise=Unoise(n_min=-0.1, n_max=0.1),
    clip=(-1.0, 1.0),
)
```

Height scanner 配置：
```python
from isaaclab.sensors import RayCasterCfg, patterns

height_scanner = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/Trunk",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
    attach_yaw_only=True,
    pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.6, 0.6]),  # 7x7=49 rays
    debug_vis=False,
    mesh_prim_paths=["/World/ground"],
)
```

### 5.3 奖励权重调优指南

| 阶段 | 调整方向 |
|------|---------|
| 初期（0-2000 iters） | 降低能耗惩罚权重（torque/vel/acc），策略先学会站起来和跟踪速度 |
| 中期 | 逐步提高姿态惩罚（flat_orientation, lin_vel_z），提升稳定性 |
| 后期 | 加入 terrain curriculum 后微调 track 权重，使策略不过度保守 |

---

## 6. 终止条件设计

### 6.1 终止项

```python
@configclass
class TerminationsCfg:
    # 1. 超时（episode 正常结束，不算失败）
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 2. 基座倾斜过大
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 0.8},  # ~46°
    )

    # 3. 基座高度过低（摔倒）
    root_height_below_minimum = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"asset_cfg": SceneEntityCfg("robot"), "minimum_height": 0.35},
    )

    # 4. 足部以外的身体接触地面（膝盖/躯干/手触地即终止）
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "Trunk",
                    "Left_Hip_Pitch", "Right_Hip_Pitch",
                    "Left_Shank", "Right_Shank",
                    "left_hand_link", "right_hand_link",
                ],
            ),
            "threshold": 10.0,
        },
    )
```

### 6.2 设计说明

| 条件 | 阈值 | 理由 |
|------|------|------|
| `bad_orientation` | 0.8 rad (~46°) | 人形机器人倾斜超过 45° 基本无法自行恢复 |
| `root_height_below_minimum` | 0.35 m | K1 正常站立时 Trunk 高度约 0.57m，低于 0.35m 意味着摔倒或蹲伏失败 |
| `illegal_contact` | 10 N | 膝盖、躯干、手掌触地即判失败 |

---

## 7. 地形配置

### 7.1 阶段一：平地

```python
# FlatEnvCfg — 纯平地，训练基础步态
self.scene.terrain.terrain_type = "plane"
```

### 7.2 阶段二：粗糙地形

使用 `TerrainGeneratorCfg`，混合多种地形类型：

```python
self.scene.terrain.terrain_type = "generator"
self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=5,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,  # ✅ 启用课程学习
    sub_terrains={
        # 40% 平地/轻微纹理
        "nearly_flat": HfRandomUniformTerrainCfg(
            proportion=0.4,
            noise_range=(0.0, 0.008),     # 0-0.8cm 波动
            noise_step=0.005,
            border_width=0.25,
        ),
        # 20% 缓坡 (±5°~15°)
        "pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.2,
            slope_range=(0.05, 0.25),      # ~3°~14° 坡度
            platform_width=1.5,
            border_width=0.25,
        ),
        # 15% 低台阶 (2-5cm)
        "discrete_obstacles": HfDiscreteObstaclesTerrainCfg(
            proportion=0.15,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.1, 0.4),
            obstacle_height_range=(0.02, 0.05),  # 2-5cm 台阶
            num_obstacles=8,
            platform_width=1.0,
            border_width=0.25,
        ),
        # 15% 波浪地形
        "wave": HfWaveTerrainCfg(
            proportion=0.15,
            amplitude_range=(0.01, 0.04),  # 1-4cm 波幅
            num_waves=3,
            border_width=0.25,
        ),
        # 10% 随机粗糙
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.10,
            noise_range=(-0.02, 0.02),     # ±2cm 随机凹凸
            noise_step=0.005,
            border_width=0.25,
        ),
    },
)
```

### 7.3 地形类型及其训练目标

| 地形 | 占比 | 难度 | 训练目标 |
|------|------|------|---------|
| `nearly_flat` | 40% | ★☆ | 基础步态、速度跟踪、站立 |
| `pyramid_slope` | 20% | ★★ | 上坡/下坡时的姿态调节和步长适应 |
| `discrete_obstacles` | 15% | ★★★ | 抬脚高度、落脚精度、越障能力 |
| `wave` | 15% | ★★ | 不平地面上的动态平衡 |
| `random_rough` | 10% | ★★ | 随机扰动的泛化能力 |

### 7.4 课程学习

```python
curriculum=True  # Isaac Lab 自动按 difficulty 参数递增地形难度
```

Isaac Lab 的 `TerrainGenerator` 在 `curriculum=True` 时，会根据每个 sub-terrain 的 `difficulty` 参数，按 episode 数逐步从简单地形过渡到困难地形。需要在 sub-terrain 配置中添加 `difficulty` 参数。

---

## 8. 域随机化与扰动

### 8.1 启动时随机化（不变）

沿用现有 `fight_001` 的随机化设置：

```python
@configclass
class EventCfg:
    # 物理材质随机化
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 0.6),
            "dynamic_friction_range": (0.3, 0.6),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )
    # 关节默认位置噪声
    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip.*", ".*_Knee.*", ".*_Ankle.*"]),
            "pos_distribution_params": (-0.02, 0.02),
            "operation": "add",
        },
    )
    # 质心随机化
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.06, 0.06), "z": (-0.06, 0.06)},
        },
    )
```

### 8.2 新增随机化

```python
    # 身体质量随机化（增强鲁棒性）
    randomize_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (-0.05, 0.05),  # ±5% 质量变化
            "operation": "scale",
        },
    )
```

### 8.3 外力扰动（间隔触发）

```python
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.5, 4.0),  # 每 1.5-4 秒推一次
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-0.5, 0.5),
            },
        },
    )
```

---

## 9. 课程学习策略

### 9.1 两阶段训练

| 阶段 | Iterations | 地形 | 环境配置 |
|------|-----------|------|---------|
| **Stage I — Flat** | 0 ~ 10,000 | 纯平面 (`terrain_type="plane"`) | `FlatEnvCfg` |
| **Stage II — Rough** | 10,000 ~ 50,000 | 混合地形 (`terrain_type="generator"`) + 地形 curriculum | `RoughEnvCfg`（从 Stage I checkpoint 恢复） |

### 9.2 Stage I 目标

- 学会稳定站立（不摔倒）
- 在所有速度指令下产生基本步态
- 前进速度跟踪误差 < 0.2 m/s

### 9.3 Stage II 目标

- 在坡面（最高 ~14°）上稳定行走
- 跨越 2-5cm 台阶
- 在波浪地面上保持平衡
- 推进速度跟踪误差 < 0.15 m/s（即使在粗糙地形上）

---

## 10. 机器人配置

### 10.1 上肢关节锁定

在 `BOOSTER_K1_CFG` 基础上创建 locomotion 专用配置，将上半身关节刚度设为 0（位置控制模式下锁死在默认位置）：

```python
# 方法一：在 ActionCfg 中仅对下肢关节生效
@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            ".*_Hip_Pitch",
            ".*_Hip_Roll",
            ".*_Hip_Yaw",
            ".*_Knee_Pitch",
            ".*_Ankle_Pitch",
            ".*_Ankle_Roll",
        ],
        use_default_offset=True,
    )
```

这样策略只输出 12 维动作（下肢关节），上肢关节保持在 URDF 中定义的默认位置不动。

### 10.2 接触传感器

只需监控下肢相关 body：

```python
contact_forces = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*",
    history_length=3,
    track_air_time=True,
    force_threshold=10.0,
    debug_vis=False,
)
```

---

## 11. PPO 超参数

```python
@configclass
class PPORunnerCfg(BasePPORunnerCfg):
    max_iterations = 50000
    experiment_name = "k1_locomotion"
    num_steps_per_env = 24
    save_interval = 1000

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,             # 初始探索噪声（locomotion 适当降低）
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,              # locomotion 通常需要更高熵（探索多种步态）
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```

### 与 Beyond Mimic 的主要差异

| 参数 | Beyond Mimic | Locomotion | 理由 |
|------|-------------|------------|------|
| `init_noise_std` | 1.0 | 0.8 | locomotion 动作空间更小（12 vs 22），适当降低噪声 |
| `entropy_coef` | 0.005 | 0.01 | locomotion 需要更多探索来发现自然步态 |
| `max_iterations` | 50,000 | 50,000 | 第一阶段够用，后续可增加 |

---

## 12. 训练与评估命令

### 12.1 训练

```bash
# Stage I — 平地训练
python scripts/rsl_rl/train.py \
    --task Booster-K1-Locomotion-Flat-v0 \
    --headless \
    --device cuda:0 \
    --num_envs 4096 \
    --max_iterations 10000

# Stage II — 粗糙地形训练（从 Stage I checkpoint 恢复）
python scripts/rsl_rl/train.py \
    --task Booster-K1-Locomotion-Rough-v0 \
    --headless \
    --device cuda:0 \
    --num_envs 4096 \
    --max_iterations 50000 \
    --resume \
    --load_run <stage1_log_dir>
```

### 12.2 评估/推理

```bash
python scripts/rsl_rl/play.py \
    --task Booster-K1-Locomotion-Flat-v0-Play \
    --checkpoint <path_to_checkpoint>
```

### 12.3 列出所有可用环境

```bash
python scripts/list_envs.py
```

---

## 13. 实现步骤

### Step 1: 创建 locomotion MDP 模块（不修改已有文件）

- [ ] 可选：在 `mdp/` 下创建 `locomotion_rewards.py`（如需自定义奖励如 feet_air_time）
- [ ] 可选：在 `mdp/` 下创建 `locomotion_observations.py`（如需自定义观测）

> **原则**：优先复用 Isaac Lab 内置函数（`mdp.track_lin_vel_xy_exp`、`mdp.base_lin_vel` 等），只在必要时才写自定义函数。这样最小化新增代码量。

### Step 2: 创建 `tracking_env_cfg.py`

- [ ] 定义 `CommandsCfg` — 使用 `UniformVelocityCommandCfg`
- [ ] 定义 `ActionsCfg` — 仅下肢 12 关节
- [ ] 定义 `ObservationsCfg` — 48 维观测
- [ ] 定义 `RewardsCfg` — 13 项奖励
- [ ] 定义 `TerminationsCfg` — 超时 + 倾斜 + 高度 + 非法接触
- [ ] 定义 `EventCfg` — 域随机化 + 外力扰动
- [ ] 定义 `TrackingEnvCfg` 基类

### Step 3: 创建 `env_cfg.py`

- [ ] `FlatEnvCfg(TrackingEnvCfg)` — 平地
- [ ] `RoughEnvCfg(FlatEnvCfg)` — 混合地形
- [ ] `PlayFlatEnvCfg(FlatEnvCfg)` — 推理（play 模式 + 无外力）

### Step 4: 创建 `ppo_cfg.py`

- [ ] 继承 `BasePPORunnerCfg`，调整超参数

### Step 5: 创建 `__init__.py`

- [ ] 注册 3 个 Gym 环境：
  - `Booster-K1-Locomotion-Flat-v0` → `FlatEnvCfg`
  - `Booster-K1-Locomotion-Rough-v0` → `RoughEnvCfg`
  - `Booster-K1-Locomotion-Flat-v0-Play` → `PlayFlatEnvCfg`

### Step 6: 验证

- [ ] 运行 `python scripts/list_envs.py` 确认 3 个环境正确注册
- [ ] 用少量 envs（如 `--num_envs 64`）运行 100 步确认无 crash
- [ ] 开始 Stage I 训练
- [ ] 在 TensorBoard 中监控奖励曲线和速度跟踪误差

---

## 14. 预期里程碑

### Stage I 成功标准（Flat，~10k iters）

| 指标 | 目标值 |
|------|--------|
| 存活率（episode 不因摔倒终止） | > 95% |
| 前进速度跟踪误差 (vx error) | < 0.15 m/s |
| 侧移速度跟踪误差 (vy error) | < 0.12 m/s |
| 转向角速度跟踪误差 (ωz error) | < 0.2 rad/s |
| 平均 episode 奖励 | > 15 |

### Stage II 成功标准（Rough，~50k iters）

| 指标 | 目标值 |
|------|--------|
| 存活率 | > 90%（含地形扰动） |
| 速度跟踪误差 | 与 Stage I 持平或更好 |
| 坡面行走成功率 | > 80%（坡度 ~14°） |
| 台阶越障成功率 | > 70%（2-5cm 台阶） |

---

## 附录 A: K1 下肢关节参考

```
Left_Hip_Pitch      Right_Hip_Pitch       # 髋俯仰（前后摆腿）
Left_Hip_Roll        Right_Hip_Roll        # 髋侧摆（左右摆腿）
Left_Hip_Yaw         Right_Hip_Yaw         # 髋旋转（内外旋）
Left_Knee_Pitch      Right_Knee_Pitch      # 膝俯仰
Left_Ankle_Pitch     Right_Ankle_Pitch     # 踝俯仰
Left_Ankle_Roll      Right_Ankle_Roll      # 踝侧摆
```

## 附录 B: 参考实现

Isaac Lab 已有完整的人形机器人 locomotion 实现，可作为代码模板：

| 机器人 | 文件路径 |
|--------|---------|
| **H1** (平地和粗糙) | `isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/h1/rough_env_cfg.py` |
| **G1** (平地和粗糙) | `isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py` |
| **Locomotion 基础类** | `isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py` |
| **Locomotion 专用 MDP** | `isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py` |

这些文件位于 Isaac Lab 安装目录下的 `source/isaaclab_tasks/` 中。

**关键发现**：
- H1/G1 使用 `track_lin_vel_xy_yaw_frame_exp`（yaw-aligned 坐标系下的速度跟踪），比 base-frame 更适合人形机器人
- H1/G1 使用 `track_ang_vel_z_world_exp`（世界坐标系下的角速度跟踪）
- 有专用的 `feet_air_time_positive_biped`（双足正奖励，仅在单支撑相给奖励）
- 有 `feet_slide` 惩罚（惩罚脚在接触地面时的滑动）
- 有 `stand_still_joint_deviation_l1`（仅在站立指令时惩罚关节偏离默认姿态）

## 附录 C: Isaac Lab 内置依赖一览

本方案直接使用的 Isaac Lab 内置组件：

| 类别 | 使用的类/函数 | 来源模块 |
|------|-------------|---------|
| 命令 | `UniformVelocityCommand`, `UniformVelocityCommandCfg` | `isaaclab.envs.mdp.commands.velocity_command` |
| 观测 | `base_lin_vel`, `base_ang_vel`, `projected_gravity`, `joint_pos_rel`, `joint_vel_rel`, `last_action`, `generated_commands` | `isaaclab.envs.mdp.observations` |
| 奖励 | `track_lin_vel_xy_exp`, `track_ang_vel_z_exp`, `lin_vel_z_l2`, `ang_vel_xy_l2`, `flat_orientation_l2`, `base_height_l2`, `joint_torques_l2`, `joint_vel_l2`, `joint_acc_l2`, `action_rate_l2`, `joint_pos_limits`, `undesired_contacts` | `isaaclab.envs.mdp.rewards` |
| 终止 | `time_out`, `bad_orientation`, `root_height_below_minimum`, `illegal_contact` | `isaaclab.envs.mdp.terminations` |
| 事件 | `randomize_rigid_body_material`, `randomize_joint_default_pos`, `randomize_rigid_body_com`, `randomize_rigid_body_mass`, `push_by_setting_velocity` | `isaaclab.envs.mdp.events` |
| 地形 | `HfRandomUniformTerrainCfg`, `HfPyramidSlopedTerrainCfg`, `HfDiscreteObstaclesTerrainCfg`, `HfWaveTerrainCfg` | `isaaclab.terrains.height_field.hf_terrains_cfg` |
