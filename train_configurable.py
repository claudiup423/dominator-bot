"""
Configurable PPO training — reads config from a JSON file or CLI args.

The web app's "Start Training" button generates a config JSON and writes it
to data/train_config.json, then this script picks it up and starts training.

Can also be run directly:
    python train_configurable.py --config data/train_config.json
    python train_configurable.py --mode 1v1 --checkpoint data/checkpoints/.../37558116

Config JSON schema:
{
    "mode": "1v1" | "2v2" | "3v3",
    "checkpoint_path": null | "path/to/checkpoint/step_dir",
    "rewards": {
        "goal": 10.0,
        "touch": 3.0,
        "velocity_ball_to_goal": 5.0,
        "velocity_player_to_ball": 1.0,
        "speed": 0.1,
        "boost_penalty": 0.0,
        "demo": 0.0,
        "aerial": 0.0
    },
    "hyperparameters": {
        "policy_lr": 5e-4,
        "critic_lr": 5e-4,
        "n_proc": 16,
        "ppo_batch_size": 50000,
        "ts_per_iteration": 50000,
        "ppo_epochs": 3,
        "ppo_ent_coef": 0.01,
        "gamma": 0.99,
        "tick_skip": 8
    },
    "training": {
        "save_every_ts": 5000000,
        "timestep_limit": 10000000000,
        "timeout_seconds": 15,
        "log_to_wandb": false
    },
    "run_id": "uuid-from-web-app"  // optional, for API integration
}
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

def _create_env():
    import json
    config_path = os.environ["DOMINATOR_ACTIVE_CONFIG"]
    with open(config_path) as f:
        config = json.load(f)
    return build_env(config)

# ─── Default config ──────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "mode": "1v1",
    "checkpoint_path": None,
    "rewards": {
        "goal": 10.0,
        "touch": 3.0,
        "velocity_ball_to_goal": 5.0,
        "velocity_player_to_ball": 1.0,
        "speed": 0.1,
        "boost_penalty": 0.0,
        "demo": 0.0,
        "aerial": 0.0,
    },
    "hyperparameters": {
        "policy_lr": 5e-4,
        "critic_lr": 5e-4,
        "n_proc": 16,
        "ppo_batch_size": 50_000,
        "ts_per_iteration": 50_000,
        "ppo_epochs": 3,
        "ppo_ent_coef": 0.01,
        "gamma": 0.99,
        "tick_skip": 8,
    },
    "training": {
        "save_every_ts": 5_000_000,
        "timestep_limit": 10_000_000_000,
        "timeout_seconds": 15,
        "log_to_wandb": False,
    },
    "run_id": None,
}


# ─── Reward functions ────────────────────────────────────────────────
# We import lazily so the config can be parsed without rlgym installed

def build_reward_functions(reward_config: dict):
    """Build a CombinedReward from the config dict."""
    from rlgym.api import RewardFunction
    from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
    from rlgym.rocket_league import common_values

    class VelocityPlayerToBallReward(RewardFunction):
        def reset(self, agents, initial_state, shared_info): pass
        def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
            rewards = {}
            for agent in agents:
                car = state.cars[agent]
                ball = state.ball
                car_to_ball = ball.position - car.physics.position
                dist = np.linalg.norm(car_to_ball)
                direction = car_to_ball / (dist + 1e-8)
                vel_toward = np.dot(car.physics.linear_velocity, direction)
                rewards[agent] = vel_toward / 2300.0
            return rewards

    class VelocityBallToGoalReward(RewardFunction):
        def reset(self, agents, initial_state, shared_info): pass
        def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
            rewards = {}
            for agent in agents:
                car = state.cars[agent]
                goal = np.array(
                    common_values.ORANGE_GOAL_CENTER if car.team_num == common_values.BLUE_TEAM
                    else common_values.BLUE_GOAL_CENTER
                )
                ball_to_goal = goal - state.ball.position
                direction = ball_to_goal / (np.linalg.norm(ball_to_goal) + 1e-8)
                vel_toward = np.dot(state.ball.linear_velocity, direction)
                rewards[agent] = vel_toward / 6000.0
            return rewards

    class SpeedReward(RewardFunction):
        def reset(self, agents, initial_state, shared_info): pass
        def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
            rewards = {}
            for agent in agents:
                speed = np.linalg.norm(state.cars[agent].physics.linear_velocity)
                rewards[agent] = speed / 2300.0
            return rewards

    class BoostPenaltyReward(RewardFunction):
        def reset(self, agents, initial_state, shared_info): pass
        def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
            rewards = {}
            for agent in agents:
                # Penalize using boost when above 50
                boost = state.cars[agent].boost_amount
                rewards[agent] = -max(0, boost - 0.5)
            return rewards

    class DemoReward(RewardFunction):
        def reset(self, agents, initial_state, shared_info):
            self._prev_demos = {}
        def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
            rewards = {}
            for agent in agents:
                car = state.cars[agent]
                rewards[agent] = 1.0 if car.is_demoed else 0.0
            return rewards

    class AerialReward(RewardFunction):
        def reset(self, agents, initial_state, shared_info): pass
        def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
            rewards = {}
            for agent in agents:
                car = state.cars[agent]
                height = car.physics.position[2]
                # Reward being in the air with ball contact
                rewards[agent] = max(0, (height - 300) / 2000.0) if not car.on_ground else 0.0
            return rewards

    # Build reward list from config
    reward_map = {
        "goal": (GoalReward, {}),
        "touch": (TouchReward, {}),
        "velocity_ball_to_goal": (VelocityBallToGoalReward, {}),
        "velocity_player_to_ball": (VelocityPlayerToBallReward, {}),
        "speed": (SpeedReward, {}),
        "boost_penalty": (BoostPenaltyReward, {}),
        "demo": (DemoReward, {}),
        "aerial": (AerialReward, {}),
    }

    components = []
    for name, weight in reward_config.items():
        if weight > 0 and name in reward_map:
            cls, kwargs = reward_map[name]
            components.append((cls(**kwargs), weight))

    if not components:
        # Fallback: at least use goal + touch
        components = [(GoalReward(), 10.0), (TouchReward(), 1.0)]

    return CombinedReward(*components)


def build_env(config: dict):
    """Build rlgym environment from config."""
    from rlgym.api import RLGym
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, AnyCondition
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
    from rlgym_ppo.util import RLGymV2GymWrapper

    mode = config["mode"]
    hyper = config["hyperparameters"]
    training = config["training"]

    # Team sizes
    sizes = {"1v1": (1, 1), "2v2": (2, 2), "3v3": (3, 3)}
    blue, orange = sizes.get(mode, (1, 1))

    tick_skip = hyper.get("tick_skip", 8)
    timeout = training.get("timeout_seconds", 15)

    action_parser = RepeatAction(LookupTableAction(), repeats=tick_skip)
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=timeout),
    )
    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue, orange_size=orange),
        KickoffMutator(),
    )
    obs_builder = DefaultObs(zero_padding=None)
    reward_fn = build_reward_functions(config["rewards"])
    engine = RocketSimEngine()

    env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        transition_engine=engine,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
    )
    return RLGymV2GymWrapper(env)


def find_checkpoints() -> list[dict]:
    """Scan for available checkpoints and return metadata."""
    base = Path("data/checkpoints")
    found = []
    if not base.exists():
        return found

    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        for step_dir in sorted(run_dir.iterdir()):
            if not step_dir.is_dir() or not step_dir.name.isdigit():
                continue
            policy_file = step_dir / "PPO_POLICY.pt"
            if policy_file.exists():
                found.append({
                    "path": str(step_dir),
                    "step": int(step_dir.name),
                    "run": run_dir.name,
                    "size_mb": round(policy_file.stat().st_size / 1024 / 1024, 1),
                })

    # Also check _v1 backup
    base_v1 = Path("data/checkpoints_v1")
    if base_v1.exists():
        for run_dir in sorted(base_v1.iterdir()):
            if not run_dir.is_dir():
                continue
            for step_dir in sorted(run_dir.iterdir()):
                if not step_dir.is_dir() or not step_dir.name.isdigit():
                    continue
                policy_file = step_dir / "PPO_POLICY.pt"
                if policy_file.exists():
                    found.append({
                        "path": str(step_dir),
                        "step": int(step_dir.name),
                        "run": run_dir.name + " (v1)",
                        "size_mb": round(policy_file.stat().st_size / 1024 / 1024, 1),
                    })

    return sorted(found, key=lambda x: x["step"], reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Configurable PPO Training")
    parser.add_argument("--config", type=str, default="data/train_config.json",
                        help="Path to config JSON file")
    parser.add_argument("--mode", type=str, choices=["1v1", "2v2", "3v3"], default=None)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    parser.add_argument("--list-checkpoints", action="store_true",
                        help="List available checkpoints and exit")
    parser.add_argument("--generate-config", action="store_true",
                        help="Generate a default config file and exit")
    args = parser.parse_args()

    # List checkpoints mode
    if args.list_checkpoints:
        cps = find_checkpoints()
        if not cps:
            print("No checkpoints found.")
        else:
            print(f"Found {len(cps)} checkpoints:\n")
            for cp in cps:
                print(f"  Step {cp['step']:>12,}  |  {cp['size_mb']:>5.1f} MB  |  {cp['path']}")
        return

    # Generate config mode
    if args.generate_config:
        config_path = Path(args.config)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        print(f"Default config written to {config_path}")
        return

    # Load config
    config = DEFAULT_CONFIG.copy()
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            user_config = json.load(f)
        # Deep merge
        for key in user_config:
            if isinstance(user_config[key], dict) and key in config:
                config[key] = {**config[key], **user_config[key]}
            else:
                config[key] = user_config[key]
        print(f"Loaded config from {config_path}")
    else:
        print(f"No config at {config_path}, using defaults")

    # CLI overrides
    if args.mode:
        config["mode"] = args.mode
    if args.checkpoint:
        config["checkpoint_path"] = args.checkpoint

    # Print config
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"  Mode:          {config['mode']}")
    print(f"  Checkpoint:    {config['checkpoint_path'] or 'Fresh start'}")
    print(f"  Rewards:")
    for name, weight in config["rewards"].items():
        if weight > 0:
            print(f"    {name:30s} = {weight}")
    print(f"  Hyperparameters:")
    for name, val in config["hyperparameters"].items():
        print(f"    {name:30s} = {val}")
    print(f"  Training:")
    for name, val in config["training"].items():
        print(f"    {name:30s} = {val}")
    print("=" * 60 + "\n")

    # Save active config for reference
    active_path = Path("data/metrics/active_config.json")
    active_path.parent.mkdir(parents=True, exist_ok=True)
    with open(active_path, "w") as f:
        json.dump(config, f, indent=2)

    # Build environment
    hyper = config["hyperparameters"]
    training = config["training"]

    from rlgym_ppo import Learner

    os.environ["DOMINATOR_ACTIVE_CONFIG"] = str(active_path.resolve())

    learner_kwargs = dict(
        env_create_function=_create_env,
        policy_layer_sizes=[256, 256],
        critic_layer_sizes=[256, 256],
        ppo_batch_size=hyper.get("ppo_batch_size", 50_000),
        ts_per_iteration=hyper.get("ts_per_iteration", 50_000),
        exp_buffer_size=hyper.get("ppo_batch_size", 50_000) * 3,
        ppo_minibatch_size=hyper.get("ppo_batch_size", 50_000) // 2,
        ppo_ent_coef=hyper.get("ppo_ent_coef", 0.01),
        ppo_epochs=hyper.get("ppo_epochs", 3),
        policy_lr=hyper.get("policy_lr", 5e-4),
        critic_lr=hyper.get("critic_lr", 5e-4),
        n_proc=hyper.get("n_proc", 16),
        standardize_returns=True,
        standardize_obs=False,
        save_every_ts=training.get("save_every_ts", 5_000_000),
        timestep_limit=training.get("timestep_limit", 10_000_000_000),
        log_to_wandb=training.get("log_to_wandb", False),
        render=False,
        add_unix_timestamp=True,
    )

    # Resume from checkpoint if specified
    if config["checkpoint_path"]:
        cp_path = Path(config["checkpoint_path"])
        if cp_path.exists():
            learner_kwargs["checkpoint_load_folder"] = str(cp_path)
            print(f"Will resume from checkpoint: {cp_path}")
        else:
            print(f"WARNING: Checkpoint path not found: {cp_path}")
            print("Starting fresh instead.")

    learner = Learner(**learner_kwargs)
    learner.learn()


if __name__ == "__main__":
    main()