from isaaclab.utils import configclass
from isaaclab.terrains import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
from booster_assets import BOOSTER_ASSETS_DIR
from booster_train.assets.robots.booster import BOOSTER_K1_CFG as ROBOT_CFG, K1_ACTION_SCALE
from booster_train.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from .tracking_env_cfg import TrackingEnvCfg


@configclass
class FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = K1_ACTION_SCALE
        self.commands.motion.motion_file = f"{BOOSTER_ASSETS_DIR}/motions/K1/indian_dance/indian_dance.npz"
        self.commands.motion.anchor_body_name = "Trunk"
        self.commands.motion.body_names = [
            'Trunk',
            'Head_2',
            'Left_Hip_Roll',
            'Left_Shank',
            'left_foot_link',
            'Right_Hip_Roll',
            'Right_Shank',
            'right_foot_link',
            'Left_Arm_2',
            'Left_Arm_3',
            'left_hand_link',
            'Right_Arm_2',
            'Right_Arm_3',
            'right_hand_link',
        ]


@configclass
class FlatWoStateEstimationEnvCfg(FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class RoughWoStateEstimationEnvCfg(FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.debug_vis = False        # 设为True可视化地形分布
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(10.0, 10.0),            # 每个地形块尺寸（米）
            border_width=20.0,            # 边界宽度（米）
            num_rows=5,                   # 地形网格行数
            num_cols=10,                  # 地形网格列数
            horizontal_scale=0.1,         # 水平分辨率
            vertical_scale=0.005,         # 垂直分辨率
            slope_threshold=0.75,         # 网格简化阈值
            use_cache=False,              # 每次重新生成地形
            curriculum=False,              # 启用课程学习
            sub_terrains={
                # 80%接近平面的地形（非常平滑）
                "nearly_flat": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.8,
                    noise_range=(0.0, 0.005),    # 高度波动0-0.5cm（几乎平坦）
                    noise_step=0.005,            # 噪声步长0.5cm
                    border_width=0.25,
                ),
                # 20%随机粗糙地形
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.2,
                    noise_range=(-0.015, 0.015),    # 高度波动±1.5cm
                    noise_step=0.005,               # 噪声步长0.5cm
                    border_width=0.25,
                ),
            },
        )


# ---------------------------------------------------------------------------
# 带完整观测的配置（方案一）
# ---------------------------------------------------------------------------
# 以下两个类绕过 FlatWoStateEstimationEnvCfg，直接继承 FlatEnvCfg，
# 保留了 motion_anchor_pos_b 和 base_lin_vel 观测。
#
# 原因：
#   FlatWoStateEstimationEnvCfg 删除了这两个观测，导致 Policy 无法感知
#   "目标位置在哪" 和 "自己移动多快"，从而无法跟踪全局位置——
#   对于印度舞这种带有大幅平移的动作尤其致命（手部动作好但机器人位置跟不上）。
#
#   使用完整观测训练后，Policy 能直接获得位置误差信号，
#   训练时重点观察 Rewards/motion_global_anchor_pos 曲线是否收敛。
#
# 代价：
#   这两个观测依赖状态估计（里程计/IMU），实机部署时需要对应的估计器。
# ---------------------------------------------------------------------------


@configclass
class RoughEnvCfg(FlatEnvCfg):
    """
    训练用配置：粗糙地形 + 完整观测。

    与 RoughWoStateEstimationEnvCfg 的区别：
    - 继承 FlatEnvCfg（保留 motion_anchor_pos_b + base_lin_vel）
    - 其余配置相同：80% 近平面 + 20% 颠簸地形、push 扰动、域随机化
    """
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.debug_vis = False        # 设为True可视化地形分布
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(10.0, 10.0),            # 每个地形块尺寸（米）
            border_width=20.0,            # 边界宽度（米）
            num_rows=5,                   # 地形网格行数
            num_cols=10,                  # 地形网格列数
            horizontal_scale=0.1,         # 水平分辨率
            vertical_scale=0.005,         # 垂直分辨率
            slope_threshold=0.75,         # 网格简化阈值
            use_cache=False,              # 每次重新生成地形
            curriculum=False,              # 启用课程学习
            sub_terrains={
                # 80%接近平面的地形（非常平滑）
                "nearly_flat": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.8,
                    noise_range=(0.0, 0.005),    # 高度波动0-0.5cm（几乎平坦）
                    noise_step=0.005,            # 噪声步长0.5cm
                    border_width=0.25,
                ),
                # 20%随机粗糙地形
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.2,
                    noise_range=(-0.015, 0.015),    # 高度波动±1.5cm
                    noise_step=0.005,               # 噪声步长0.5cm
                    border_width=0.25,
                ),
            },
        )


@configclass
class PlayFlatEnvCfg(FlatEnvCfg):
    """
    推理/演示用配置：完全平面 + 完整观测。

    与 PlayFlatWoStateEstimationEnvCfg 的区别：
    - 继承 FlatEnvCfg（保留 motion_anchor_pos_b + base_lin_vel）
    - 其余配置相同：motion 顺序播放、无 push 扰动
    """
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.play = True
        self.events.push_robot = None


# ---------------------------------------------------------------------------
# 原始 WoStateEstimation 配置（保留备用）
# ---------------------------------------------------------------------------


@configclass
class PlayFlatWoStateEstimationEnvCfg(FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.play = True
        self.events.push_robot = None
