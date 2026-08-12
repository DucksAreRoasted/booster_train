"""Behavior tests for local RSL-RL command-line overrides."""

import argparse
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

CLI_ARGS_PATH = Path(__file__).parents[1] / "scripts/rsl_rl/cli_args.py"
TRAIN_PATH = Path(__file__).parents[1] / "scripts/rsl_rl/train.py"


def _load_cli_args_module():
    spec = importlib.util.spec_from_file_location("booster_rsl_rl_cli_args", CLI_ARGS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _agent_cfg(*, load_optimizer: bool):
    return SimpleNamespace(
        seed=42,
        resume=False,
        load_run=".*",
        load_checkpoint="model_.*.pt",
        run_name="",
        logger="tensorboard",
        wandb_project="isaaclab",
        neptune_project="isaaclab",
        load_optimizer=load_optimizer,
    )


def test_cli_can_explicitly_retain_optimizer_state_when_resuming():
    """Users can override a task's clean-optimizer transfer default."""
    cli_args = _load_cli_args_module()
    parser = argparse.ArgumentParser()
    cli_args.add_rsl_rl_args(parser)
    args = parser.parse_args(["--load_optimizer"])

    cfg = cli_args.update_rsl_rl_cfg(_agent_cfg(load_optimizer=False), args)

    assert cfg.load_optimizer is True


def test_cli_preserves_the_task_optimizer_default_when_unspecified():
    """Omitting optimizer flags leaves each task's transfer policy intact."""
    cli_args = _load_cli_args_module()
    parser = argparse.ArgumentParser()
    cli_args.add_rsl_rl_args(parser)
    args = parser.parse_args([])

    cfg = cli_args.update_rsl_rl_cfg(_agent_cfg(load_optimizer=False), args)

    assert cfg.load_optimizer is False


def test_cli_can_explicitly_reset_optimizer_state_when_resuming():
    """Existing tasks can request a clean optimizer without changing their config."""
    cli_args = _load_cli_args_module()
    parser = argparse.ArgumentParser()
    cli_args.add_rsl_rl_args(parser)
    args = parser.parse_args(["--reset_optimizer"])

    cfg = cli_args.update_rsl_rl_cfg(_agent_cfg(load_optimizer=True), args)

    assert cfg.load_optimizer is False


def test_training_honors_the_configured_optimizer_resume_policy():
    """The training entry point forwards the public optimizer setting to RSL-RL."""
    module = ast.parse(TRAIN_PATH.read_text())
    load_call = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "runner.load"
    )
    load_optimizer = next(keyword.value for keyword in load_call.keywords if keyword.arg == "load_optimizer")

    assert ast.unparse(load_optimizer) == "agent_cfg.load_optimizer"
