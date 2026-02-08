"""
Watch the PPO-trained bot play.

Usage:
    python watch_bot.py              # Try rlviser 3D viewer
    python watch_bot.py --headless   # Print actions to terminal (no viewer needed)
"""

import numpy as np
import torch
import os
import sys
import time

from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, AnyCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import (
    MutatorSequence,
    FixedTeamSizeMutator,
    KickoffMutator,
)
from rlgym.rocket_league import common_values
from rlgym_ppo.util import RLGymV2GymWrapper


def find_latest_checkpoint():
    checkpoint_base = "data/checkpoints"
    if not os.path.exists(checkpoint_base):
        print("ERROR: No data/checkpoints directory found.")
        sys.exit(1)

    best_step = -1
    best_path = None
    for dirname in os.listdir(checkpoint_base):
        full = os.path.join(checkpoint_base, dirname)
        if not os.path.isdir(full):
            continue
        for sub in os.listdir(full):
            subpath = os.path.join(full, sub)
            if os.path.isdir(subpath) and sub.isdigit():
                step = int(sub)
                if step > best_step:
                    best_step = step
                    best_path = subpath

    if best_path is None:
        print("ERROR: No checkpoints found.")
        sys.exit(1)

    return best_path, best_step


def load_policy(checkpoint_dir):
    policy_path = os.path.join(checkpoint_dir, "PPO_POLICY.pt")
    policy = torch.nn.Sequential(
        torch.nn.Linear(172, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 90),
    )
    state_dict = torch.load(policy_path, map_location="cpu")
    cleaned = {k.replace("model.", ""): v for k, v in state_dict.items()}
    policy.load_state_dict(cleaned)
    policy.eval()
    return policy


def build_env(use_rlviser=True):
    team_size = 3
    tick_skip = 8

    action_parser = RepeatAction(LookupTableAction(), repeats=tick_skip)
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=30),
    )

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=team_size, orange_size=team_size),
        KickoffMutator(),
    )

    obs_builder = DefaultObs(zero_padding=None)
    reward_fn = CombinedReward((GoalReward(), 1.0),)
    engine = RocketSimEngine()

    renderer = None
    if use_rlviser:
        try:
            from rlgym.rocket_league.rlviser import RLViserRenderer
            renderer = RLViserRenderer()
            print("RLViser renderer enabled.")
        except Exception as e:
            print(f"RLViser not available: {e}")
            print("Running headless.")

    env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        transition_engine=engine,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        renderer=renderer,
    )

    return env


def run_headless(policy, env):
    """Run without viewer, print stats."""
    obs_dict = env.reset()
    agents = list(obs_dict.keys())

    goals_blue = 0
    goals_orange = 0
    touches = 0
    steps = 0
    episodes = 0

    print(f"\nHeadless mode — {len(agents)} agents. Press Ctrl+C to stop.\n")
    print(f"{'Step':>8} | {'Blue':>4} - {'Orange':<4} | {'Touches':>7} | {'Action Distribution':>30}")
    print("-" * 75)

    action_counts = np.zeros(90)

    try:
        while True:
            actions = {}
            for agent in agents:
                obs = obs_dict[agent]
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits = policy(obs_tensor)
                    action = torch.argmax(logits, dim=-1).item()
                actions[agent] = np.array([action])
                action_counts[action] += 1

            obs_dict, reward_dict, terminated_dict, truncated_dict = env.step(actions)
            steps += 1

            # Check for goals from rewards
            for agent, reward in reward_dict.items():
                if reward > 0.5:  # GoalReward = 1.0
                    car = env.state.cars[agent] if hasattr(env, 'state') else None
                    goals_blue += 1  # Approximate

            if steps % 100 == 0:
                top5_actions = np.argsort(action_counts)[-5:][::-1]
                top5_str = ", ".join([f"{a}({int(action_counts[a])})" for a in top5_actions])
                print(f"{steps:>8} | {goals_blue:>4} - {goals_orange:<4} | {touches:>7} | {top5_str}")

            if any(terminated_dict.values()) or any(truncated_dict.values()):
                episodes += 1
                obs_dict = env.reset()
                agents = list(obs_dict.keys())
                print(f"  >> Episode {episodes} ended. Resetting...")

    except KeyboardInterrupt:
        print(f"\n\nStopped after {steps} steps, {episodes} episodes.")
        print(f"Most used actions: {np.argsort(action_counts)[-10:][::-1].tolist()}")


def run_visual(policy, env):
    """Run with rlviser viewer."""
    obs_dict = env.reset()
    agents = list(obs_dict.keys())

    print(f"\nVisual mode — {len(agents)} agents. Close rlviser window to stop.\n")

    try:
        while True:
            actions = {}
            for agent in agents:
                obs = obs_dict[agent]
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits = policy(obs_tensor)
                    action = torch.argmax(logits, dim=-1).item()
                actions[agent] = np.array([action])

            obs_dict, reward_dict, terminated_dict, truncated_dict = env.step(actions)
            env.render()
            time.sleep(0.008)

            if any(terminated_dict.values()) or any(truncated_dict.values()):
                obs_dict = env.reset()
                agents = list(obs_dict.keys())

    except KeyboardInterrupt:
        print("\nStopped.")


def main():
    headless = "--headless" in sys.argv

    checkpoint_dir, step = find_latest_checkpoint()
    print(f"Loading checkpoint: {checkpoint_dir} (step {step:,})")

    policy = load_policy(checkpoint_dir)
    print("Policy loaded OK.")

    # Quick sanity check
    obs = torch.randn(1, 172)
    with torch.no_grad():
        logits = policy(obs)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * probs.log()).sum().item()
    print(f"Policy entropy: {entropy:.3f} (4.5 = random, <3.0 = learned something)")

    if headless:
        env = build_env(use_rlviser=False)
        run_headless(policy, env)
    else:
        env = build_env(use_rlviser=True)
        run_visual(policy, env)


if __name__ == "__main__":
    main()
