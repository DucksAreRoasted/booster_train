"""
命令模块 (Command Module)
==========================
本模块定义了 Beyond Mimic 任务中的 **命令系统**，是整个 MDP 的核心部分。

什么是「命令」(Command)？
--------------------------
在强化学习中，「命令」告诉机器人当前要完成什么目标。
在这个任务中，命令来自于预录的动作捕捉（MoCap, Motion Capture）数据——
就像给机器人看一段"示范视频"，机器人需要模仿这段视频中的动作。

本模块包含三个主要部分：
1. **MotionLoader**   — 负责从磁盘加载 .npz 运动数据文件
2. **MotionCommand**  — 核心命令类，在每个仿真步更新运动目标
3. **MotionCommandCfg** — 配置类，用于在 env_cfg.py 中配置上述两个类的参数

数据流概览：
------------
    .npz 文件 → MotionLoader（加载并索引）→ MotionCommand（按时间步取出、计算误差）→ Policy
"""

# ---------------------------------------------------------------------------
# Python 标准库和类型注解
# ---------------------------------------------------------------------------
from __future__ import annotations  # 允许前向引用类型注解，例如在方法中使用类本身作为类型提示

import math        # 数学函数，如 log、sqrt 等
import numpy as np # NumPy，用于读取 .npz 文件（磁盘上的运动数据）
import os          # 操作系统接口，用于检查文件路径是否存在
import torch       # PyTorch，在 GPU/CPU 上做张量运算的核心库
from collections.abc import Sequence  # Sequence 类型，表示 list/tuple 等序列类型
from dataclasses import MISSING       # MISSING 是一个特殊标记（哨兵值），表示"该配置项必须由用户手动填写，没有默认值"
from typing import TYPE_CHECKING      # TYPE_CHECKING 是一个在运行时为 False 的常量，用于条件导入

# ---------------------------------------------------------------------------
# Isaac Lab 框架相关导入
# ---------------------------------------------------------------------------
from isaaclab.assets import Articulation  # Articulation 类，代表仿真中的一个关节机器人（含关节、身体等）
from isaaclab.managers import CommandTerm, CommandTermCfg  # CommandTerm 是命令的基类，CommandTermCfg 是命令配置的基类
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # 可视化标记，用于在仿真中显示调试信息
from isaaclab.markers.config import FRAME_MARKER_CFG  # 预定义的坐标系标记配置（显示 x/y/z 轴的颜色箭头）
from isaaclab.utils import configclass  # @configclass 装饰器，将一个普通类转变为 Isaac Lab 的配置类
from isaaclab.utils.math import (       # 导入各种四元数和数学工具函数
    quat_apply,           # 用四元数旋转一个向量： v_rotated = q * v * q^-1
    quat_error_magnitude, # 计算两个四元数之间的"角度差"有多大（返回标量，单位是弧度）
    quat_from_euler_xyz,  # 从欧拉角（roll, pitch, yaw）创建四元数
    quat_inv,             # 计算四元数的逆（共轭），四元数旋转"反向"
    quat_mul,             # 四元数乘法：q_result = q1 * q2（先做 q2 旋转，再做 q1 旋转）
    sample_uniform,       # 在给定范围内均匀随机采样
    yaw_quat,             # 从一个完整四元数中只提取偏航角（yaw，绕Z轴旋转）对应的四元数
)

# ---------------------------------------------------------------------------
# TYPE_CHECKING 条件导入：
# 下面这个 import 仅在 IDE/类型检查器（如 Pylance）分析代码时生效，
# 在运行时不会真正导入，避免循环引用问题。
# ---------------------------------------------------------------------------
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv  # 强化学习环境的主类，只在类型注解中使用


# ===========================================================================
# MotionLoader — 运动数据加载器
# ===========================================================================
class MotionLoader:
    """
    运动数据加载器（Motion Data Loader）

    负责从磁盘上的 .npz 文件加载运动捕捉数据，并提供按索引访问的接口。

    什么是 .npz 文件？
    -------------------
    .npz 是 NumPy 的压缩数组格式。它像一个"字典"，里面可以存多个 numpy 数组。
    在 beyond_mimic 中，.npz 文件通常由运动预处理管线（motion preprocessing pipeline）
    生成——将原始 MoCap(动作捕捉) 数据经重定向/IK/仿真验证后打包，包含了机器人运动捕捉的逐帧数据。

    文件内部结构（字段名）：
        - "fps":             float, 运动数据的帧率（frames per second）
        - "body_names":      str 列表，身体部位名称，如 ["pelvis", "left_foot", "right_foot", ...]
        - "joint_names":     str 列表，关节名称，如 ["hip_yaw", "hip_roll", "hip_pitch", ...]
        - "joint_pos":       形状为 (T, num_joints) 的 float 数组，每帧每个关节的角度（弧度）
        - "joint_vel":       形状为 (T, num_joints) 的 float 数组，每帧每个关节的角速度（弧度/秒）
        - "body_pos_w":      形状为 (T, num_bodies, 3) 的 float 数组，每帧每个身体部位在世界坐标系中的 (x, y, z) 位置
        - "body_quat_w":     形状为 (T, num_bodies, 4) 的 float 数组，每帧每个身体部位在世界坐标系中的朝向四元数 (w, x, y, z)
        - "body_lin_vel_w":  形状为 (T, num_bodies, 3) 的 float 数组，每帧每个身体部位在世界坐标系中的线速度
        - "body_ang_vel_w":  形状为 (T, num_bodies, 3) 的 float 数组，每帧每个身体部位在世界坐标系中的角速度
        其中 T = 总帧数

    坐标系约定：
        _w 后缀 = world frame（世界坐标系），以仿真世界的原点为参考
        _b 后缀 = body frame（身体局部坐标系），以某个身体部位为参考

    参数说明：
    ----------
    motion_file : str
        .npz 文件的路径。例如 "motions/walk.npz"
    track_body_names : Sequence[str]
        我们需要跟踪（模仿）的身体部位名称列表。
        例如 ["pelvis", "left_foot", "right_foot"]
        注意：这些名称必须出现在 .npz 的 body_names 中
    track_joint_names : Sequence[str]
        我们需要跟踪的关节名称列表。
        例如 ["hip_yaw", "hip_roll", "hip_pitch", ...]
        注意：这些名称必须出现在 .npz 的 joint_names 中
    default_motion_body_names : Sequence[str] | None
        如果 .npz 文件中缺少 "body_names" 字段，则使用此备选列表。
        通常设置为机器人 URDF 中的 body 名称列表
    default_motion_joint_names : Sequence[str] | None
        如果 .npz 文件中缺少 "joint_names" 字段，则使用此备选列表。
        通常设置为机器人 URDF 中的 joint 名称列表
    tail_len : int
        轨迹尾部的"安全缓冲区"，单位是帧。
        采样起始帧时，不会从最后 tail_len 帧中采样，避免轨迹播放到末尾时突然中断。
        默认为 0（无缓冲）
    device : str
        数据存储在哪个设备上，"cpu" 或 "cuda:0"。
        如果设置为 GPU，后续所有运算都在 GPU 上进行，速度更快。
    """

    def __init__(self, motion_file: str,
                 track_body_names: Sequence[str],
                 track_joint_names: Sequence[str],
                 *,
                 default_motion_body_names: Sequence[str] | None = None,
                 default_motion_joint_names: Sequence[str] | None = None,
                 tail_len: int = 0, device: str = "cpu"):
        # --- 步骤1：校验文件路径是否存在 ---
        # assert 是"断言"，如果表达式为 False，程序会立即报错并停止。
        # 这是一种"尽早失败"的编程习惯，避免在后续步骤中发现数据不存在。
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"

        # --- 步骤2：从磁盘加载 .npz 文件 ---
        # np.load() 返回一个类似于字典的对象，用 data["key"] 的方式访问每个字段
        data = np.load(motion_file)

        # FPS: 帧率，例如 30 表示每秒 30 帧，每帧间隔 1/30 ≈ 0.033 秒
        self.fps = data["fps"]

        # --- 步骤3：获取身体部位名称（body_names）---
        # 优先使用 .npz 文件中存储的名称，如果没有则使用备选列表
        if "body_names" in data:
            self._body_names = data["body_names"].tolist()  # numpy数组 → Python列表
        else:
            assert default_motion_body_names is not None, \
                "Motion file missing body_names, and no default_body_names provided."
            self._body_names = default_motion_body_names

        # --- 步骤4：获取关节名称（joint_names）---
        if "joint_names" in data:
            self._joint_names = data["joint_names"].tolist()
        else:
            assert default_motion_joint_names is not None, \
                "Motion file missing joint_names, and no default_joint_names provided."
            self._joint_names = default_motion_joint_names

        # --- 步骤5：建立索引映射 ---
        # 数据文件中可能包含很多身体/关节（全部人体关节），但我们只需跟踪其中一部分。
        # 例如数据有 30 个身体，机器人只需要模仿 5 个，我们就只取这 5 个的索引。
        # self._body_names.index(name) 返回 name 在列表中的位置（0, 1, 2, ...）
        # 结果：self._body_indexes = [3, 15, 22] 表示我们需要的身体在数据的第3、15、22列
        self._body_indexes = torch.tensor(
            [self._body_names.index(name) for name in track_body_names],
            dtype=torch.long,   # long = int64，用于索引
            device=device
        )
        self._joint_indexes = torch.tensor(
            [self._joint_names.index(name) for name in track_joint_names],
            dtype=torch.long,
            device=device
        )

        # --- 步骤6：将运动数据从 NumPy 数组转换为 PyTorch 张量 ---
        # 这样数据就可以在 GPU 上并行运算。
        # 形状说明：data["joint_pos"] 形状为 (T, num_joints_in_file)
        # 我们只取需要的关节列 [:, self._joint_indexes]
        # 结果形状：(T, num_tracked_joints)
        self.joint_pos = torch.tensor(
            data["joint_pos"], dtype=torch.float32, device=device
        )[:, self._joint_indexes]  # 在所有帧中只选跟踪的关节

        self.joint_vel = torch.tensor(
            data["joint_vel"], dtype=torch.float32, device=device
        )[:, self._joint_indexes]

        # 身体数据也是同理，但我们保存完整数据（所有身体），
        # 通过 @property 按索引提取，这样更灵活
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)

        # 总帧数 T：运动数据一共有多少帧
        self.time_step_total = self.joint_pos.shape[0]

        # 尾部缓冲长度（帧数），采样时不会取最后 tail_len 帧
        self.tail_len = tail_len

    # ------------------------------------------------------------------
    # @property — Python 装饰器，让方法像属性一样访问
    # 例如 motion.body_pos_w  而不是 motion.body_pos_w()
    # 下面的四个属性返回"我们需要跟踪的身体部位"的数据子集
    # ------------------------------------------------------------------

    @property
    def body_pos_w(self) -> torch.Tensor:
        """返回我们需要跟踪的身体部位在世界坐标系中的位置。形状: (T, num_tracked_bodies, 3)"""
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        """返回我们需要跟踪的身体部位在世界坐标系中的朝向四元数。形状: (T, num_tracked_bodies, 4)"""
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """返回我们需要跟踪的身体部位在世界坐标系中的线速度。形状: (T, num_tracked_bodies, 3)"""
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """返回我们需要跟踪的身体部位在世界坐标系中的角速度。形状: (T, num_tracked_bodies, 3)"""
        return self._body_ang_vel_w[:, self._body_indexes]

    @property
    def max_reset_frame(self) -> int:
        """
        最大可重置帧索引。

        为什么需要这个？
        ---------------
        当运动播放到末尾时，我们需要"重新开始"——从某帧开始播。
        但我们不能从最后 tail_len 帧开始，因为那样运动马上就要结束了。
        所以 max_reset_frame = 总帧数 - tail_len，保证采样到的起始帧后面还有足够多的帧。

        例如：总帧数 = 1000, tail_len = 200，则 max_reset_frame = 800
        表示我们可以从第 0 到 799 帧中随机选一帧作为起点。
        """
        return self.time_step_total - self.tail_len


# ===========================================================================
# MotionCommand — 运动跟踪命令
# ===========================================================================
class MotionCommand(CommandTerm):
    """
    运动跟踪命令（Motion Tracking Command）

    这是整个 Beyond Mimic 任务的核心类。它实现了以下功能：
    1. 每个仿真步推进运动时间 → 告诉机器人"当前目标是什么"
    2. 当运动播放完毕时重新采样起始帧 → 保持训练持续进行
    3. 基于失败率的自适应采样 → 让训练更集中于困难的动作片段
    4. 计算跟踪误差指标 → 用于日志记录和调试

    继承自 CommandTerm（Isaac Lab 框架中所有命令的基类）。
    框架在每步仿真自动调用以下方法：
        - _update_command()  每个仿真步调用一次
        - _resample_command()  当 episode 重置时调用
        - _update_metrics()   用于记录到 TensorBoard 等

    「锚点」(Anchor) 的概念：
        Anchor 是所有位姿计算的"参照物"。我们选择机器人/运动数据中的某个身体部位
        （如 pelvis/骨盆）作为锚点，所有其他身体部位的位置和朝向都相对于锚点来表达。
        这样做的好处是：策略学的是"身体部位之间如何协调"，而不被全局坐标所干扰。

    参数：
    -------
    cfg : MotionCommandCfg
        命令配置对象，包含所有可配置参数
    env : ManagerBasedRLEnv
        强化学习环境，包含仿真场景、机器人、管理器等
    """

    # cfg 的类型注解，IDE 和类型检查器会使用这个信息
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):

        # --- 步骤1：调用父类初始化（CommandTerm 基类）---
        # 父类会设置 self._env, self.num_envs, self.device 等常用属性
        super().__init__(cfg, env)

        # --- 步骤2：获取仿真中的机器人对象 ---
        # env.scene 包含了场景中的所有物体（机器人、地面、障碍物等）
        # cfg.asset_name 是在配置文件中指定的名字，如 "robot"
        self.robot: Articulation = env.scene[cfg.asset_name]

        # --- 步骤3：找到锚点身体的索引 ---
        # 机器人方面：anchor_body_name 是锚点的名称（如 "pelvis"），
        # 在机器人的身体名称列表中查找它的索引
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        # 打印机器人的所有身体名称，方便调试
        print(f'{self.robot.body_names=}')

        # 运动数据方面：同样找到锚点在运动数据 body_names 列表中的位置
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)

        # --- 步骤4：获取需要跟踪的身体部位在机器人中的索引 ---
        # robot.find_bodies() 返回 (body_indices, body_names) 的元组
        # preserve_order=True 保证返回顺序与输入顺序一致
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
            dtype=torch.long, device=self.device
        )

        # --- 步骤5：确定备用身体/关节名称 ---
        # "or" 运算符：如果左边是 None 或空列表，则使用右边的默认值
        # 这些备选列表用于 .npz 文件缺少 body_names/joint_names 字段时
        default_motion_body_names = self.cfg.default_motion_body_names or self.robot.body_names
        default_motion_joint_names = self.cfg.default_motion_joint_names or self.robot.joint_names

        # --- 步骤6：创建运动数据加载器 ---
        # MotionLoader 从磁盘加载 .npz 文件到 GPU/CPU 内存
        self.motion = MotionLoader(
            self.cfg.motion_file,       # .npz文件路径
            self.cfg.body_names,        # 要跟踪的身体部位
            self.robot.joint_names,     # 要跟踪的关节（用机器人的关节名来匹配数据中的关节）
            default_motion_body_names=default_motion_body_names,
            default_motion_joint_names=default_motion_joint_names,
            tail_len=self.cfg.tail_len, # 轨迹尾部缓冲区
            device=self.device          # GPU 或 CPU
        )

        # --- 步骤7：初始化每个并行环境的当前时间步 ---
        # self.num_envs 是并行环境数量（通常 4096 或更多）
        # 每个环境独立维护自己的运动播放进度：time_steps[i] = 环境 i 当前播放到第几帧
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # --- 步骤8：初始化"相对位姿"缓冲区 ---
        # body_pos_relative_w: 每个身体部位相对于锚点的目标位置
        # body_quat_relative_w: 每个身体部位相对于锚点的目标朝向（四元数）
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        # 四元数的 w 分量初始化为 1（表示"无旋转"的恒等四元数: (1, 0, 0, 0)）
        self.body_quat_relative_w[:, :, 0] = 1.0

        # --- 步骤9：初始化自适应采样相关参数 ---
        # 自适应采样是什么？
        # 有些动作片段很难（比如后空翻），机器人经常在这些片段失败。
        # 我们希望多采样这些困难片段来加强训练，少采样简单片段（比如站立不动）。
        #
        # 将整个运动轨迹分成 bin_count 个"桶"（bins），每个桶包含若干帧。
        # 当某个桶对应的帧播放后导致 episode 失败时，该桶的"失败计数"增加。
        # 下次采样起始帧时，失败多的桶有更高的概率被选中。
        #
        # 桶数 = 轨迹可采样帧数 / 每个 episode 的帧数
        # 每个 episode 的帧数 = 1 / (decimation * sim.dt)
        #   - sim.dt: 物理仿真步长（如 0.005 秒）
        #   - decimation: 策略决策频率（如 10，表示每 10 个物理步做一次策略决策）
        #   - 因此 1/(decimation*sim.dt) = 每秒策略决策次数
        self.bin_count = int(self.motion.max_reset_frame // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1

        # bin_failed_count: 每个桶的"平滑失败计数"（指数滑动平均）
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        # _current_bin_failed: 当前 episode 每个桶的失败计数（原始值，未平滑）
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)

        # 自适应采样用的卷积核（kernel）
        # kernel = [λ^0, λ^1, ..., λ^{k-1}] 归一化后
        # 例如 λ=0.8, k=3 → kernel = [1, 0.8, 0.64] / 2.44 = [0.41, 0.33, 0.26]
        # 这个核会与失败计数做一维卷积，使得相邻的桶也会获得一定的采样概率提升
        # （因为一个困难的帧附近往往也是困难的，平滑化让采样更合理）
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
            device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()  # 归一化，使得核的和 = 1

        # --- 步骤10：初始化监控指标（metrics）---
        # metrics 字典来自父类 CommandTerm，所有值都会被记录到 TensorBoard
        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    # ==================================================================
    # 命令属性：返回"目标值"（Reference / Target）
    # 这些值描述的是运动数据中"应该达到"的状态
    # ==================================================================

    @property
    def command(self) -> torch.Tensor:
        """
        命令向量 —— 直接给 Policy（策略网络）作为输入。
        当前实现：将目标关节位置和速度拼接在一起。
        形状: (num_envs, num_joints * 2)
        TODO: 这是最佳观测吗？也许可以被更好的表示替代。
        """
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        """
        目标关节位置。从运动数据中按当前时间步取出。
        形状: (num_envs, num_joints)
        每个环境的时间步不同（time_steps[i]），所以取出的帧也不同。
        """
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        """目标关节速度。形状: (num_envs, num_joints)"""
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        """
        目标身体部位在世界坐标系中的位置。
        需要加上 env_origins（每个并行环境的原点偏移）以对齐到仿真世界。
        形状: (num_envs, num_bodies, 3)
        """
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        """目标身体部位在世界坐标系中的朝向四元数。形状: (num_envs, num_bodies, 4)"""
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """目标身体部位在世界坐标系中的线速度。形状: (num_envs, num_bodies, 3)"""
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """目标身体部位在世界坐标系中的角速度。形状: (num_envs, num_bodies, 3)"""
        return self.motion.body_ang_vel_w[self.time_steps]

    # ---- 锚点（Anchor）属性：单点的状态，是所有相对计算的参照物 ----

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        """
        目标锚点的世界位置。
        锚点是所有位姿的参考中心，通常选择 pelvis（骨盆）。
        形状: (num_envs, 3)
        """
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        """目标锚点的世界朝向。形状: (num_envs, 4)"""
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        """目标锚点的世界线速度。形状: (num_envs, 3)"""
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        """目标锚点的世界角速度。形状: (num_envs, 3)"""
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    # ==================================================================
    # 机器人状态属性：返回"实际值"（Measured / Actual）
    # 这些值从仿真中读取机器人的实际状态
    # ==================================================================

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        """机器人实际关节位置（从仿真中读取）。形状: (num_envs, num_joints)"""
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        """机器人实际关节速度。形状: (num_envs, num_joints)"""
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        """机器人实际身体部位在世界坐标系中的位置。形状: (num_envs, num_tracked_bodies, 3)"""
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        """机器人实际身体部位在世界坐标系中的朝向。形状: (num_envs, num_tracked_bodies, 4)"""
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        """机器人实际身体部位在世界坐标系中的线速度。形状: (num_envs, num_tracked_bodies, 3)"""
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        """机器人实际身体部位在世界坐标系中的角速度。形状: (num_envs, num_tracked_bodies, 3)"""
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        """机器人锚点在世界坐标系中的实际位置。形状: (num_envs, 3)"""
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        """机器人锚点在世界坐标系中的实际朝向。形状: (num_envs, 4)"""
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        """机器人锚点在世界坐标系中的实际线速度。形状: (num_envs, 3)"""
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        """机器人锚点在世界坐标系中的实际角速度。形状: (num_envs, 3)"""
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    # ==================================================================
    # _update_metrics — 更新监控指标
    # 框架在每个仿真步自动调用此方法，计算结果被记录到 TensorBoard
    # ==================================================================
    def _update_metrics(self):
        """
        计算并更新所有跟踪误差指标。

        所有指标都是"目标值（motion data）− 实际值（robot）"的误差。
        - torch.norm(x, dim=-1): 计算向量的 L2 范数（欧几里得距离），沿最后一维求
        - quat_error_magnitude(q1, q2): 计算两个四元数之间的角度差（弧度）
        - .mean(dim=-1): 对所有身体部位求平均（如果有多个被跟踪的身体部位）
        """

        # ---- 锚点误差（全局坐标系） ----
        self.metrics["error_anchor_pos"] = torch.norm(
            self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
        )
        self.metrics["error_anchor_rot"] = quat_error_magnitude(
            self.anchor_quat_w, self.robot_anchor_quat_w
        )
        self.metrics["error_anchor_lin_vel"] = torch.norm(
            self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
        )
        self.metrics["error_anchor_ang_vel"] = torch.norm(
            self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
        )

        # ---- 身体部位误差（相对坐标系） ----
        # 对所有跟踪的身体部位求平均
        self.metrics["error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)

        self.metrics["error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(dim=-1)

        # ---- 身体速度误差（全局坐标系） ----
        self.metrics["error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)

        self.metrics["error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)

        # ---- 关节误差 ----
        self.metrics["error_joint_pos"] = torch.norm(
            self.joint_pos - self.robot_joint_pos, dim=-1
        )
        self.metrics["error_joint_vel"] = torch.norm(
            self.joint_vel - self.robot_joint_vel, dim=-1
        )

    # ==================================================================
    # _adaptive_sampling — 自适应采样（核心训练策略）
    # ==================================================================
    def _adaptive_sampling(self, env_ids: Sequence[int]):
        """
        自适应采样算法 — 让训练更集中于困难的动作片段。

        整体流程：
        ----------
        1. 检查哪些环境刚因为失败而重置了 episode
        2. 在这些失败环境采样到的 bin 上增加"失败计数"
        3. 对失败计数做一维卷积平滑（让相邻 bin 也获得一些提升）
        4. 加上 uniform 噪声（保证即使从未失败的 bin 也有最低采样概率）
        5. 归一化得到采样概率分布
        6. 按此分布采样新的起始帧

        这个设计的直觉：
        - 如果机器人总是在"转身"这段动作摔倒，
          那"转身"对应的 bin 会积累更多的失败计数，
          下次开始新 episode 时，"转身"被选中的概率更大，
          机器人有更多机会练习这段动作。
        """

        # ---- 步骤1：检查哪些环境刚失败了 ----
        # self._env.termination_manager.terminated 是一个布尔数组
        # True 表示该环境的 episode 因违反终止条件而结束
        episode_failed = self._env.termination_manager.terminated[env_ids]

        if torch.any(episode_failed):  # 如果至少有一个环境失败了
            # 将失败环境当前的时间步映射到对应的 bin
            # 公式：bin_index = time_step * bin_count / max_reset_frame
            # torch.clamp(..., 0, bin_count-1): 确保索引不越界
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.max_reset_frame, 1),
                0, self.bin_count - 1
            )
            # 提取出那些失败环境的 bin 索引
            fail_bins = current_bin_index[env_ids][episode_failed]
            # torch.bincount: 统计每个 bin 出现了多少次失败
            # 例如 fail_bins = [2, 2, 5, 5, 5] → bincount = [0,0,2,0,0,3,...]
            self._current_bin_failed[:] = torch.bincount(
                fail_bins, minlength=self.bin_count
            )

        # ---- 步骤2：构建采样概率分布 ----
        # 基础概率 = 平滑失败计数 + 均匀噪声
        # uniform_ratio: 均匀噪声的比例，保证了最小采样概率（exploration）
        # 例如 uniform_ratio=0.1 表示至少 10% 的概率来自均匀分布
        sampling_probabilities = (
            self.bin_failed_count
            + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        )

        # ---- 步骤3：用一维卷积平滑概率分布 ----
        # 为什么需要平滑？
        # 如果一个 bin 很困难，它的邻居通常也困难（困难帧周围也是困难帧）。
        # 如果不平滑，采样可能过于集中在少数几帧，导致过拟合。
        #
        # 具体操作：
        # 1. unsqueeze: 将 (bin_count,) 重塑为 (1, 1, bin_count) 以适应 conv1d 输入格式
        # 2. pad: 在右侧补零，使用 "replicate" 模式（边界复制）
        # 3. conv1d: 用指数衰减核做一维卷积
        # 4. view(-1): 展平回 (bin_count,)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),    # (bin_count,) → (1, 1, bin_count)
            (0, self.cfg.adaptive_kernel_size - 1),              # 只在右侧 pad
            mode="replicate",                                     # 边界值复制
        )
        sampling_probabilities = torch.nn.functional.conv1d(
            sampling_probabilities,                              # 输入
            self.kernel.view(1, 1, -1),                          # 卷积核 (1, 1, kernel_size)
        ).view(-1)                                                # 展平

        # ---- 步骤4：归一化为概率分布 ----
        # 所有概率加起来 = 1
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        # ---- 步骤5：按多项式分布采样 ----
        # torch.multinomial: 按给定概率分布采样，replacement=True 表示可重复采样
        sampled_bins = torch.multinomial(
            sampling_probabilities, len(env_ids), replacement=True
        )

        # ---- 步骤6：将 bin 索引转换为时间步 ----
        # 每个 bin 覆盖若干帧，在 bin 内均匀随机选择具体帧
        # 公式：time_step = bin_index / bin_count * max_reset_frame
        # 加一个小随机偏移避免采样到 bin 的边界
        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.max_reset_frame - 1)
        ).long()
        # 上面的两行代码做了同样的事，第二行覆盖了第一行（这是一种冗余的写法）
        self.time_steps[env_ids] = (
            sampled_bins / self.bin_count * (self.motion.max_reset_frame - 1)
        ).long()

        # ---- 步骤7：记录采样相关的监控指标 ----
        # 熵（Entropy）：衡量采样分布的"均匀性"
        # H = -∑ p_i * log(p_i)
        # H=0 表示总是从同一个 bin 采样（极端集中）
        # H=1 表示所有 bin 被等概率采样（完全均匀）
        # 归一化熵 = H / log(bin_count)，映射到 [0, 1] 区间
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)

        # 最高概率的 bin 及其概率
        pmax, imax = sampling_probabilities.max(dim=0)

        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    # ==================================================================
    # _resample_command — 重置/重新采样命令
    # 当运动播放完毕或 episode 开始时调用
    # ==================================================================
    def _resample_command(self, env_ids: Sequence[int]):
        """
        重新采样运动命令。在以下情况被调用：
        1. Episode 刚开始时（所有环境都需要一个起始帧）
        2. 运动播放到末尾时（需要"重播"）

        这个方法做了两件关键的事：
        - 使用自适应采样选择新的起始帧
        - 将机器人的状态重置到起始帧对应的状态（位置+速度+姿态）
          注意：会加入少量随机噪声，增强策略的鲁棒性
        """

        # 如果没有需要采样的环境，直接返回
        if len(env_ids) == 0:
            return

        # ---- 步骤1：自适应采样选择起始帧 ----
        self._adaptive_sampling(env_ids)

        # ---- 步骤2：如果是"播放"模式，从头开始播放 ----
        if self.cfg.play:
            self.time_steps[env_ids] = 0

        # ---- 步骤3：提取起始帧的根部（root）状态 ----
        # root 指的是运动数据的第一个被跟踪的身体部位（索引为 0），通常是 pelvis
        # .clone() 创建副本，避免修改原始数据
        root_pos = self.body_pos_w[:, 0].clone()       # 位置 (T, 3)
        root_ori = self.body_quat_w[:, 0].clone()      # 朝向 (T, 4)
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()  # 线速度 (T, 3)
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()  # 角速度 (T, 3)

        # ---- 步骤4：对根部位置和朝向加入随机扰动 ----
        # pose_range 指定了每个自由度的随机范围
        # 例如 {"x": (-0.1, 0.1), "yaw": (-0.5, 0.5)}
        # 这样做的好处：策略见过各种不同的起始状态，泛化能力更强
        range_list = [
            self.cfg.pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        # 在指定范围内均匀采样随机偏移量
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        # 给位置加入随机偏移
        root_pos[env_ids] += rand_samples[:, 0:3]
        # 从随机欧拉角创建四元数，然后乘以原始朝向
        orientations_delta = quat_from_euler_xyz(
            rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
        )
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])

        # ---- 步骤5：对根部速度也加入随机扰动 ----
        range_list = [
            self.cfg.velocity_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        # ---- 步骤6：获取并随机化关节状态 ----
        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        # 对关节位置加入随机噪声
        # joint_position_range 如 (-0.52, 0.52) 弧度 ≈ (-30°, 30°)
        joint_pos += sample_uniform(
            *self.cfg.joint_position_range, joint_pos.shape, joint_pos.device
        )
        # 裁剪到机器人的关节软限位内，避免设置超出关节物理范围的值
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        # soft_joint_pos_limits 形状：(num_envs, num_joints, 2)，最后一个维度是 (下限, 上限)
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids],
            soft_joint_pos_limits[:, :, 0],  # 下限
            soft_joint_pos_limits[:, :, 1],  # 上限
        )

        # ---- 步骤7：将随机化后的状态写入仿真器 ----
        # write_joint_state_to_sim: 设置机器人的关节角度和速度
        self.robot.write_joint_state_to_sim(
            joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
        )
        # write_root_state_to_sim: 设置机器人的根部（底座）状态
        # 将位置(3) + 四元数(4) + 线速度(3) + 角速度(3) = 13 维拼接在一起
        self.robot.write_root_state_to_sim(
            torch.cat([
                root_pos[env_ids],      # (N, 3)
                root_ori[env_ids],      # (N, 4)
                root_lin_vel[env_ids],  # (N, 3)
                root_ang_vel[env_ids],  # (N, 3)
            ], dim=-1),  # 沿最后一维拼接 → (N, 13)
            env_ids=env_ids,
        )

    # ==================================================================
    # _update_command — 每个仿真步的核心更新
    # 框架在每个仿真步自动调用此方法
    # ==================================================================
    def _update_command(self):
        """
        每个仿真步调用的核心更新方法。

        做了三件事：
        1. 推进所有环境的时间步（time_steps += 1）
        2. 检查哪些环境的运动播放到了末尾，为它们重新采样起始帧
        3. 计算相对位姿（身体部位相对于锚点的位置和朝向）
        """

        # ---- 步骤1：时间步 + 1 ----
        # 所有环境的时间步向前推进一帧
        self.time_steps += 1

        # ---- 步骤2：处理"播完"的环境 ----
        # 检查哪些环境的时间步已经超出了运动数据的总帧数
        # torch.where(condition)[0] 返回满足条件的元素的索引
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        # 为这些环境重新采样起始帧，机器人也会被重置到新的起始状态
        self._resample_command(env_ids)

        # ---- 步骤3：计算相对位姿 ----
        # 我们需要计算"每个身体部位相对于锚点"的目标位姿
        # 这样做的原因：策略不应该依赖全局坐标（在房间里哪个位置），
        # 而应该学习"身体部位之间如何协调"。

        # 将锚点数据从 (N, 3/4) 扩展为 (N, num_bodies, 3/4) 以支持广播
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )

        # 机器人锚点与目标锚点之间的"位置差"
        # x 和 y 使用机器人当前锚点的位置（让策略不必跟随全局位置）
        # z（高度）使用目标锚点的高度（高度必须跟随，否则机器人会爬下或跳）
        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]

        # 机器人锚点与目标锚点之间的"偏航角差"
        # quat_mul(q1, quat_inv(q2)): 从 q2 到 q1 的旋转差
        # yaw_quat(): 只取偏航角（绕 Z 轴的旋转），忽略 roll 和 pitch
        # 这样做的：策略需要跟踪偏航角，但不需要严格模仿 roll/pitch（因为机器人可能因各种原因倾斜）
        delta_ori_w = yaw_quat(
            quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
        )

        # 计算相对于机器人的目标身体部位位姿
        # body_quat_relative_w = delta_ori * body_quat_w
        # 含义：先将身体部位朝向旋转到机器人坐标，再应用偏航差
        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)

        # body_pos_relative_w = delta_pos + rotate(body_pos_w - anchor_pos_w, delta_ori)
        # 含义：身体部位相对锚点的位置向量，经偏航差旋转后，再加上位置差
        self.body_pos_relative_w = delta_pos_w + quat_apply(
            delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
        )

        # ---- 步骤4：更新平滑的失败计数 ----
        # 使用指数滑动平均（EMA）更新失败计数
        # bin_failed_count = α * current_failed + (1-α) * bin_failed_count
        # adaptive_alpha 很小（如 0.001），意味着 bin_failed_count 变化很慢，
        # 能够积累长时间的训练历史
        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed
            + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        # 清零当前失败计数，为下一个 episode 做准备
        self._current_bin_failed.zero_()

    # ==================================================================
    # _set_debug_vis_impl — 调试可视化开/关
    # ==================================================================
    def _set_debug_vis_impl(self, debug_vis: bool):
        """
        开启或关闭调试可视化。

        当开启时，会在仿真器中显示：
        - 绿色的坐标系：目标锚点（goal anchor）和目标身体部位（goal body）
        - 白色的坐标系：机器人实际锚点（current anchor）和实际身体部位（current body）

        这有助于直观地看到机器人在模仿运动时的跟踪效果。
        第一次调用时会创建可视化标记对象，后续调用只切换可见性。
        """

        if debug_vis:
            # ---- 开启可视化 ----
            # hasattr(object, "name") 检查对象是否已有某个属性
            # 第一次调用时不存在，所以会进入 if 创建可视化对象
            if not hasattr(self, "current_anchor_visualizer"):
                # 为锚点创建可视化标记
                # VisualizationMarkers: Isaac Lab 中用于在 3D 场景画点/线/轴的类
                # .replace() 修改预定义的默认配置，指定在场景中的路径
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(
                        prim_path="/Visuals/Command/current/anchor"
                    )
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(
                        prim_path="/Visuals/Command/goal/anchor"
                    )
                )

                # 为每个被跟踪的身体部位创建可视化标记
                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(
                                prim_path="/Visuals/Command/current/" + name
                            )
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(
                                prim_path="/Visuals/Command/goal/" + name
                            )
                        )
                    )

            # 切换为可见
            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            # ---- 关闭可视化 ----
            # 注意：不删除对象，只是隐藏。下次开启时无需重新创建。
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    # ==================================================================
    # _debug_vis_callback — 每帧更新可视化标记的位置
    # ==================================================================
    def _debug_vis_callback(self, event):
        """
        在每帧渲染时调用的回调函数，更新可视化标记的位置。

        即使仿真暂停，可视化也会更新（因为渲染循环独立于物理循环）。
        visualize() 方法将对应的坐标系标记移动到指定位置和朝向。
        """
        # 如果机器人还没初始化完成，跳过（避免错误）
        if not self.robot.is_initialized:
            return

        # 更新锚点标记：显示实际 vs 目标的锚点坐标系
        self.current_anchor_visualizer.visualize(
            self.robot_anchor_pos_w, self.robot_anchor_quat_w
        )
        self.goal_anchor_visualizer.visualize(
            self.anchor_pos_w, self.anchor_quat_w
        )

        # 更新身体部位标记：逐个身体更新
        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(
                self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i]
            )
            self.goal_body_visualizers[i].visualize(
                self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i]
            )


# ===========================================================================
# MotionCommandCfg — 运动命令的配置类
# ===========================================================================
@configclass  # @configclass 装饰器将这个类标记为 Isaac Lab 的配置类
class MotionCommandCfg(CommandTermCfg):
    """
    运动命令的配置类（Configuration Class）

    在 Isaac Lab 中，配置类用于在 Python 文件（如 env_cfg.py）中以声明式的方式
    设置所有参数，而不是在代码中硬编码。

    使用示例（在 env_cfg.py 中）：
        commands = {
            "motion": MotionCommandCfg(
                asset_name="robot",
                motion_file="/path/to/motion.npz",
                anchor_body_name="pelvis",
                body_names=["pelvis", "left_foot", "right_foot"],
                pose_range={"x": (-0.1, 0.1), "yaw": (-0.5, 0.5)},
            )
        }

    配置项详解：
    ===========
    """

    # ---- 必须由子类指定的类型字段 ----
    # class_type 告诉框架用哪个 Python 类来实例化这个命令
    class_type: type = MotionCommand

    # ---- 模式控制 ----
    # play: 如果为 True，每次重置时都从第 0 帧开始播放（用于实际部署/推演）
    #       如果为 False，每次重置时随机采样起始帧（用于训练）
    play: bool = False

    # ---- 资产和机器人相关 ----
    # asset_name: 在 env.scene 中注册的机器人名称，如 "robot"
    #   MISSING = 必须由用户在配置中明确提供，否则运行时报错
    asset_name: str = MISSING

    # ---- 运动数据文件 ----
    # motion_file: .npz 运动数据文件的路径
    #   MISSING = 必须由用户提供
    motion_file: str = MISSING

    # ---- 锚点和身体配置 ----
    # anchor_body_name: 锚点身体部位名称，所有相对坐标以此为中心
    #   通常选择 robot 的 pelvis（骨盆）或 torso（躯干）
    #   例如 "pelvis"
    anchor_body_name: str = MISSING

    # body_names: 需要跟踪的身体部位名称列表
    #   例如 ["pelvis", "left_thigh", "right_thigh", "left_foot", "right_foot"]
    body_names: list[str] = MISSING

    # default_motion_body_names: 当 .npz 文件缺少 body_names 时的备选列表
    #   设为 None 时自动使用 robot.body_names
    default_motion_body_names: list[str] | None = None

    # default_motion_joint_names: 当 .npz 文件缺少 joint_names 时的备选列表
    #   设为 None 时自动使用 robot.joint_names
    default_motion_joint_names: list[str] | None = None

    # tail_len: 轨迹尾部缓冲区（帧数），不从此范围内采样起始帧
    #   防止采样到运动末尾导致 episode 过早结束
    tail_len: int = 0

    # ---- 初始状态随机化范围 ----
    # pose_range: 初始根部（root/pelvis）位姿的随机扰动范围
    #   支持的 key: "x", "y", "z" (米), "roll", "pitch", "yaw" (弧度)
    #   例如 {"x": (-0.1, 0.1), "yaw": (-0.5, 0.5)}
    #   对于未指定的 key，默认使用 (0.0, 0.0)（无扰动）
    pose_range: dict[str, tuple[float, float]] = {}

    # velocity_range: 初始根部（root/pelvis）速度的随机扰动范围
    #   支持的 key: "x", "y", "z" (线速度), "roll", "pitch", "yaw" (角速度)
    #   格式同 pose_range
    velocity_range: dict[str, tuple[float, float]] = {}

    # joint_position_range: 初始关节位置的随机扰动范围
    #   单位：弧度。例如 (-0.52, 0.52) ≈ (-30°, 30°)
    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    # ---- 自适应采样超参数 ----
    # adaptive_kernel_size: 一维卷积核的大小（必须是奇数才能对称? 这里用的是非因果核）
    #   越大的核会使困难片段的相邻帧也有更高的采样概率
    #   默认 3
    adaptive_kernel_size: int = 3

    # adaptive_lambda: 卷积核的衰减因子（0 < λ <= 1）
    #   kernel = [λ^0, λ^1, ..., λ^{k-1}] 归一化
    #   例如 λ=0.8, k=3:
    #     kernel = [1, 0.8, 0.64] / 2.44 ≈ [0.41, 0.33, 0.26]
    #   λ 越接近 1，相邻帧获得的影响越大（更平滑）
    #   λ 越接近 0，失败计数几乎不影响相邻帧
    adaptive_lambda: float = 0.8

    # adaptive_uniform_ratio: 采样概率中"均匀噪声"的比例
    #   取值范围 [0, 1]
    #   0.0 = 完全按失败率采样（可能过度集中于困难片段）
    #   0.1 = 10% 来自均匀采样，90% 来自失败率（推荐）
    #   1.0 = 完全均匀采样（忽略失败率）
    adaptive_uniform_ratio: float = 0.1

    # adaptive_alpha: 失败计数的 EMA 更新率（0 < α <= 1）
    #   每个 episode 结束时的更新公式：
    #     failed_count = α * current_failed + (1-α) * failed_count
    #   较小的 α（如 0.001）意味着失败计数变化非常缓慢，
    #   积累了长时间的训练历史，不会被个别异常 episode 干扰
    adaptive_alpha: float = 0.001

    # ---- 可视化配置 ----
    # anchor_visualizer_cfg: 锚点标记的样式配置
    #   FRAME_MARKER_CFG 是 Isaac Lab 预定义的坐标轴标记
    #   scale 控制标记的大小（米）
    #   prim_path 指定在 USD 场景中的路径
    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    # body_visualizer_cfg: 身体部位标记的样式配置
    #   比锚点标记小一些（0.1 vs 0.2），便于区分
    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
