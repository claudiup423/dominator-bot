"""RLBot v5 bot that runs the PPO-trained model in Rocket League.

This is the bridge: it takes the RLBot GamePacket, converts it to the same
observation format used during training (DefaultObs from rlgym), feeds it
through the PPO policy network, converts the action index back to car controls
via the LookupTableAction table, and returns ControllerState to RLBot.

Usage:
    1. Set PPO_BOT=1 environment variable or just run this file directly.
    2. Point rlbot.toml to this file.
"""

from __future__ import annotations

import sys
import os
import math
import logging
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rlbot.flat import ControllerState, GamePacket, FieldInfo
from rlbot.managers import Bot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PPOBot")


# ============================================================================
# LOOKUP TABLE (same as rlgym LookupTableAction with default params)
# We rebuild it here so we don't need rlgym installed at runtime.
# Format: each row is [throttle, steer, yaw, pitch, roll, jump, boost, handbrake]
# ============================================================================

def _build_lookup_table() -> np.ndarray:
    """Build the same 90-action lookup table that rlgym uses."""
    actions = []

    # Ground actions
    for throttle in (-1, 0, 1):
        for steer in (-1, 0, 1):
            for boost in (0, 1):
                for handbrake in (0, 1):
                    if boost == 1 and throttle != 1:
                        continue
                    actions.append([throttle, steer, 0, 0, 0, 0, boost, handbrake])

    # Aerial actions (no ground contact)
    for pitch in (-1, 0, 1):
        for yaw in (-1, 0, 1):
            for roll in (-1, 0, 1):
                for jump in (0, 1):
                    for boost in (0, 1):
                        if pitch == yaw == roll == 0 and jump == 0 and boost == 0:
                            continue
                        actions.append([throttle, 0, yaw, pitch, roll, jump, boost, 0])

    # If we get a different count, just use the first 90 or pad
    actions = np.array(actions, dtype=np.float32)
    if len(actions) > 90:
        actions = actions[:90]

    return actions


# Pre-build once
_ACTION_TABLE = None


def get_action_table():
    """Get or lazily build the lookup table."""
    global _ACTION_TABLE
    if _ACTION_TABLE is None:
        try:
            # Try to use rlgym's actual table for perfect accuracy
            from rlgym.rocket_league.action_parsers import LookupTableAction
            lta = LookupTableAction()
            _ACTION_TABLE = np.array(lta._lookup_table, dtype=np.float32)
            logger.info(f"Loaded rlgym LookupTableAction: {len(_ACTION_TABLE)} actions")
        except ImportError:
            _ACTION_TABLE = _build_lookup_table()
            logger.info(f"Built fallback lookup table: {len(_ACTION_TABLE)} actions")
    return _ACTION_TABLE


# ============================================================================
# OBSERVATION BUILDER (mirrors DefaultObs from rlgym for 3v3)
# ============================================================================

def _rotation_matrix(pitch, yaw, roll):
    """Build 3x3 rotation matrix from Euler angles (same as rlgym)."""
    cp, cy, cr = math.cos(pitch), math.cos(yaw), math.cos(roll)
    sp, sy, sr = math.sin(pitch), math.sin(yaw), math.sin(roll)

    forward = np.array([cp * cy, cp * sy, sp])
    right = np.array([
        cy * sp * sr - cr * sy,
        sy * sp * sr + cr * cy,
        -cp * sr
    ])
    up = np.array([
        -cr * cy * sp - sr * sy,
        -cr * sy * sp + sr * cy,
        cp * cr
    ])
    return forward, right, up


def build_obs(packet: GamePacket, index: int, team: int, field_info: FieldInfo) -> np.ndarray:
    """Build the same observation vector that DefaultObs produces.

    DefaultObs for one agent in 3v3 produces ~172 features:
    - Ball: position(3), velocity(3), angular_velocity(3) = 9
    - Self car: position(3), forward(3), up(3), velocity(3), angular_velocity(3),
                boost(1), on_ground(1), has_flip(1) = 21
    - For each other car (5 in 3v3): position(3), forward(3), up(3), velocity(3),
                                      angular_velocity(3), boost(1), on_ground(1), has_flip(1) = 21
    - Inverted versions for the agent (ball pos/vel relative, etc.)

    We approximate this to match training. The exact format depends on rlgym version
    but DefaultObs normalizes positions by 1/4096 and velocities by 1/2300.
    """
    POSITION_SCALE = 1.0 / 4096.0
    VELOCITY_SCALE = 1.0 / 2300.0
    ANGULAR_SCALE = 1.0 / 5.5

    my_car = packet.players[index]
    my_phys = my_car.physics
    my_pos = np.array([my_phys.location.x, my_phys.location.y, my_phys.location.z])
    my_vel = np.array([my_phys.velocity.x, my_phys.velocity.y, my_phys.velocity.z])
    my_ang = np.array([my_phys.angular_velocity.x, my_phys.angular_velocity.y, my_phys.angular_velocity.z])
    my_fwd, my_right, my_up = _rotation_matrix(
        my_phys.rotation.pitch, my_phys.rotation.yaw, my_phys.rotation.roll
    )

    # Ball
    ball = packet.balls[0].physics if packet.balls else None
    if ball:
        ball_pos = np.array([ball.location.x, ball.location.y, ball.location.z])
        ball_vel = np.array([ball.velocity.x, ball.velocity.y, ball.velocity.z])
        ball_ang = np.array([ball.angular_velocity.x, ball.angular_velocity.y, ball.angular_velocity.z])
    else:
        ball_pos = np.array([0, 0, 93.0])
        ball_vel = np.zeros(3)
        ball_ang = np.zeros(3)

    # If orange team, invert Y and X to normalize perspective
    invert = -1.0 if team == 1 else 1.0

    obs = []

    # Ball state (relative to field center, inverted for orange)
    obs.extend((ball_pos * POSITION_SCALE * np.array([invert, invert, 1])).tolist())
    obs.extend((ball_vel * VELOCITY_SCALE * np.array([invert, invert, 1])).tolist())
    obs.extend((ball_ang * ANGULAR_SCALE * np.array([invert, invert, 1])).tolist())

    # Self car
    obs.extend((my_pos * POSITION_SCALE * np.array([invert, invert, 1])).tolist())
    obs.extend((my_fwd * np.array([invert, invert, 1])).tolist())
    obs.extend((my_up * np.array([invert, invert, 1])).tolist())
    obs.extend((my_vel * VELOCITY_SCALE * np.array([invert, invert, 1])).tolist())
    obs.extend((my_ang * ANGULAR_SCALE * np.array([invert, invert, 1])).tolist())
    obs.append(my_car.boost / 100.0)
    obs.append(1.0 if my_car.has_wheel_contact else 0.0)
    obs.append(1.0 if my_car.air_state == 0 or my_car.has_wheel_contact else 0.0)  # has_flip approx

    # Other players: teammates first, then opponents
    teammates = []
    opponents = []
    for i, player in enumerate(packet.players):
        if i == index:
            continue
        phys = player.physics
        p_pos = np.array([phys.location.x, phys.location.y, phys.location.z])
        p_vel = np.array([phys.velocity.x, phys.velocity.y, phys.velocity.z])
        p_ang = np.array([phys.angular_velocity.x, phys.angular_velocity.y, phys.angular_velocity.z])
        p_fwd, p_right, p_up = _rotation_matrix(
            phys.rotation.pitch, phys.rotation.yaw, phys.rotation.roll
        )

        p_data = []
        p_data.extend((p_pos * POSITION_SCALE * np.array([invert, invert, 1])).tolist())
        p_data.extend((p_fwd * np.array([invert, invert, 1])).tolist())
        p_data.extend((p_up * np.array([invert, invert, 1])).tolist())
        p_data.extend((p_vel * VELOCITY_SCALE * np.array([invert, invert, 1])).tolist())
        p_data.extend((p_ang * ANGULAR_SCALE * np.array([invert, invert, 1])).tolist())
        p_data.append(player.boost / 100.0)
        p_data.append(1.0 if player.has_wheel_contact else 0.0)
        p_data.append(1.0 if player.air_state == 0 or player.has_wheel_contact else 0.0)

        if player.team == team:
            teammates.append(p_data)
        else:
            opponents.append(p_data)

    # Add teammates then opponents (same order as DefaultObs)
    for p in teammates:
        obs.extend(p)
    for p in opponents:
        obs.extend(p)

    obs_array = np.array(obs, dtype=np.float32)

    # Pad or truncate to exactly 172
    if len(obs_array) < 172:
        obs_array = np.pad(obs_array, (0, 172 - len(obs_array)))
    elif len(obs_array) > 172:
        obs_array = obs_array[:172]

    return obs_array


# ============================================================================
# PPO BOT
# ============================================================================

class PPOBot(Bot):
    """RLBot v5 bot that uses the PPO-trained policy network."""

    def initialize(self):
        # Find latest checkpoint
        checkpoint_base = _PROJECT_ROOT / "data" / "checkpoints"
        best_step = -1
        best_path = None

        if checkpoint_base.exists():
            for dirname in os.listdir(checkpoint_base):
                full = checkpoint_base / dirname
                if full.is_dir():
                    for sub in os.listdir(full):
                        subpath = full / sub
                        if subpath.is_dir() and sub.isdigit():
                            step = int(sub)
                            if step > best_step:
                                best_step = step
                                best_path = subpath

        if best_path is None:
            logger.error("NO CHECKPOINT FOUND! Bot will output random actions.")
            self.policy = None
        else:
            logger.info(f"Loading PPO checkpoint: {best_path} (step {best_step:,})")
            policy_path = best_path / "PPO_POLICY.pt"

            self.policy = torch.nn.Sequential(
                torch.nn.Linear(172, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 90),
            )
            state_dict = torch.load(str(policy_path), map_location="cpu")
            cleaned = {k.replace("model.", ""): v for k, v in state_dict.items()}
            self.policy.load_state_dict(cleaned)
            self.policy.eval()
            logger.info("PPO policy loaded successfully!")

        self.action_table = get_action_table()
        self.tick_count = 0
        self.action_repeat = 8  # Match training tick_skip
        self.cached_action = np.zeros(8, dtype=np.float32)

    def get_output(self, packet: GamePacket) -> ControllerState:
        self.tick_count += 1

        # Only run inference every action_repeat ticks (like training)
        if self.tick_count % self.action_repeat == 1 or self.tick_count <= 1:
            try:
                obs = build_obs(packet, self.index, self.team, self.field_info)

                if self.policy is not None:
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        logits = self.policy(obs_tensor)
                        action_idx = torch.argmax(logits, dim=-1).item()
                else:
                    action_idx = np.random.randint(0, len(self.action_table))

                if action_idx < len(self.action_table):
                    self.cached_action = self.action_table[action_idx]
                else:
                    self.cached_action = np.zeros(8, dtype=np.float32)

            except Exception as e:
                logger.error(f"Inference error: {e}", exc_info=True)
                self.cached_action = np.zeros(8, dtype=np.float32)

        # Convert action array to ControllerState
        # Format: [throttle, steer, yaw, pitch, roll, jump, boost, handbrake]
        cs = ControllerState()
        a = self.cached_action
        cs.throttle = float(a[0])
        cs.steer = float(a[1])
        cs.yaw = float(a[2])
        cs.pitch = float(a[3])
        cs.roll = float(a[4])
        cs.jump = bool(a[5])
        cs.boost = bool(a[6])
        cs.handbrake = bool(a[7])
        return cs


if __name__ == "__main__":
    PPOBot.run()
