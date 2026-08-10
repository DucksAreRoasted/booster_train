# Booster 强化学习任务

## 概述

本仓库提供了一套面向 Booster 机器人的强化学习任务，基于 [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html) 构建。
目前已包含适配 Booster K1 机器人的优秀 [BeyondMimic 运动追踪](https://github.com/HybridRobotics/whole_body_tracking) 框架。
本仓库遵循标准的 Isaac Lab 项目结构，并在 IsaacLab 2.2 与 Isaac Sim 5.0 上通过了测试。

## 安装

- 按照 [安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 安装 Isaac Lab。
  推荐使用 conda 安装方式，这样可以方便地从终端调用 Python 脚本。

- 将本项目/仓库克隆或复制到 **Isaac Lab 安装目录之外**（即不要在 `IsaacLab` 目录内）：
    ```bash
    git clone https://github.com/BoosterRobotics/booster_train.git
    ```

- 下载并安装 booster_assets：
   - 克隆 [booster_assets](https://github.com/BoosterRobotics/booster_assets) 仓库，其中包含 Booster 机器人模型和运动数据。
   - 按照该仓库中的说明安装 booster_assets Python 辅助工具。

- 使用已安装 Isaac Lab 的 Python 解释器，以可编辑模式安装本库：

    ```bash
    # 如果 Isaac Lab 未安装在 Python venv 或 conda 环境中，请使用 'PATH_TO_isaaclab.sh|bat -p' 替代 'python'
    python -m pip install -e source/booster_train
    ```

- 准备 BeyondMimic 运动数据：
    ```bash
    # 如果 Isaac Lab 未安装在 Python venv 或 conda 环境中，请使用 'FULL_PATH_TO_isaaclab.sh|bat -p' 替代 'python'
    python scripts/csv_to_npz.py --headless --input_file=<BOOSTER_ASSETS路径>/motions/K1/<动作文件>.csv --input_fps=<帧率> --output_name=<BOOSTER_ASSETS路径>/motions/K1/<动作文件>.npz
    ```

## 使用方法

- 列出可用任务：

    ```bash
    # 如果 Isaac Lab 未安装在 Python venv 或 conda 环境中，请使用 'FULL_PATH_TO_isaaclab.sh|bat -p' 替代 'python'
    python scripts/list_envs.py
    ```

- 运行训练任务：

    ```bash
    # 如果 Isaac Lab 未安装在 Python venv 或 conda 环境中，请使用 'FULL_PATH_TO_isaaclab.sh|bat -p' 替代 'python'
    python scripts/rsl_rl/train.py --task=<任务名称> --headless --device cuda:N
    ```

- 运行已训练的策略并导出以用于部署：

    ```bash
    # 如果 Isaac Lab 未安装在 Python venv 或 conda 环境中，请使用 'FULL_PATH_TO_isaaclab.sh|bat -p' 替代 'python'
    python scripts/rsl_rl/play.py --task=<任务名称> --checkpoint=<检查点路径>
    ```

    此脚本还会将训练好的策略导出为 TorchScript/ONNX 文件，用于在真实机器人上部署，文件位于 `logs/rsl_rl/<实验名称>/<运行编号>/exported/`。

## 部署

模型训练并导出后，你可以使用 [booster_deploy](https://github.com/BoosterRobotics/booster_deploy) 仓库在 MuJoCo 或真实的 Booster 机器人上部署训练好的策略。更多详情请参考 [booster_deploy](https://github.com/BoosterRobotics/booster_deploy) 仓库中的说明。

## 致谢

- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)：BeyondMimic 中的运动追踪训练模块，这是一个多功能人形机器人控制框架，能够提供高动态的运动追踪能力。
