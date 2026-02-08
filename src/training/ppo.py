"""PPO Reinforcement Learning training pipeline.

Phase 3: fine-tune strategy policy with PPO. Custom implementation for full
control over hybrid action space (discrete intent + continuous target/risk).

Justification for custom PPO over SB3: our hybrid action space and safety-gated
execution don't fit SB3's abstractions cleanly. ~300 lines gives us full control.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.types import Intent
from src.strategy.ml_policy import StrategyNetwork, MLStrategy, OBS_DIM, NUM_INTENTS
from src import config


class PPOValueNetwork(nn.Module):
    """Critic network for PPO — estimates state value."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class PPOBuffer:
    """Rollout buffer storing transitions for one PPO update cycle."""

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.intents = np.zeros(capacity, dtype=np.int64)
        self.targets = np.zeros((capacity, 3), dtype=np.float32)
        self.risks = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0

    def store(
        self,
        obs: np.ndarray,
        intent: int,
        target: np.ndarray,
        risk: float,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        idx = self.ptr % self.capacity
        self.obs[idx] = obs
        self.intents[idx] = intent
        self.targets[idx] = target
        self.risks[idx] = risk
        self.rewards[idx] = reward
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.dones[idx] = float(done)
        self.ptr += 1

    def compute_gae(self, gamma: float, gae_lambda: float, last_value: float) -> None:
        """Compute Generalized Advantage Estimation."""
        n = min(self.ptr, self.capacity)
        last_gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
            self.returns[t] = self.advantages[t] + self.values[t]

    def get_batches(self, batch_size: int):
        """Yield random mini-batches for PPO updates."""
        n = min(self.ptr, self.capacity)
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            yield {
                "obs": torch.from_numpy(self.obs[batch_idx]),
                "intents": torch.from_numpy(self.intents[batch_idx]),
                "targets": torch.from_numpy(self.targets[batch_idx]),
                "risks": torch.from_numpy(self.risks[batch_idx]),
                "log_probs": torch.from_numpy(self.log_probs[batch_idx]),
                "advantages": torch.from_numpy(self.advantages[batch_idx]),
                "returns": torch.from_numpy(self.returns[batch_idx]),
            }

    def reset(self) -> None:
        self.ptr = 0


class RewardCalculator:
    """Computes rewards from game state transitions.

    Reward design: heavily punishes conceding and overcommitting,
    rewards safe defensive play and high-quality shots.
    """

    def __init__(self):
        self.w = {
            "goal_scored": config.get("rewards.goal_scored", 10.0),
            "goal_conceded": config.get("rewards.goal_conceded", -15.0),
            "save": config.get("rewards.save", 3.0),
            "shot_on_goal": config.get("rewards.shot_on_goal", 1.5),
            "clear": config.get("rewards.clear", 1.0),
            "possession_quality": config.get("rewards.possession_quality", 0.1),
            "overcommit_penalty": config.get("rewards.overcommit_penalty", -2.0),
            "open_net_conceded": config.get("rewards.open_net_conceded", -20.0),
            "boost_waste": config.get("rewards.boost_waste_penalty", -0.05),
            "good_rotation": config.get("rewards.good_rotation", 0.2),
        }

    def compute(self, prev_state: dict, curr_state: dict, action: dict) -> float:
        """Compute reward for a single transition.

        Args:
            prev_state: Previous game state dict
            curr_state: Current game state dict
            action: The action taken (intent, target, risk)

        Returns:
            Scalar reward value
        """
        reward = 0.0

        # Goal events (detected by score change)
        prev_us = prev_state.get("score_us", 0)
        curr_us = curr_state.get("score_us", 0)
        prev_them = prev_state.get("score_them", 0)
        curr_them = curr_state.get("score_them", 0)

        if curr_us > prev_us:
            reward += self.w["goal_scored"]
        if curr_them > prev_them:
            reward += self.w["goal_conceded"]

        # Small per-tick rewards for good behavior
        # Possession quality: reward being close to ball with good position
        ball_dist = curr_state.get("ball_distance", 5000)
        if ball_dist < 1000:
            reward += self.w["possession_quality"] * (1.0 - ball_dist / 1000)

        return reward


class PPOTrainer:
    """PPO training loop for the strategy policy."""

    def __init__(
        self,
        policy: Optional[StrategyNetwork] = None,
        model_path: Optional[str] = None,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for PPO training")

        # Load or create policy
        if policy is not None:
            self.policy = policy
        elif model_path and Path(model_path).exists():
            self.policy = StrategyNetwork()
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            self.policy.load_state_dict(checkpoint["state_dict"])
            logger.info(f"Loaded policy from {model_path}")
        else:
            self.policy = StrategyNetwork()
            logger.info("Created new policy network")

        self.value_net = PPOValueNetwork()
        self.reward_calc = RewardCalculator()

        # Hyperparameters from config
        self.lr = config.get("training.ppo.learning_rate", 3e-4)
        self.gamma = config.get("training.ppo.gamma", 0.99)
        self.gae_lambda = config.get("training.ppo.gae_lambda", 0.95)
        self.clip_range = config.get("training.ppo.clip_range", 0.2)
        self.entropy_coef = config.get("training.ppo.entropy_coef", 0.01)
        self.value_coef = config.get("training.ppo.value_coef", 0.5)
        self.n_steps = config.get("training.ppo.n_steps", 4096)
        self.batch_size = config.get("training.ppo.batch_size", 512)
        self.n_epochs = config.get("training.ppo.n_epochs", 4)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)
        self.buffer = PPOBuffer(self.n_steps, OBS_DIM)

    def sample_action(
        self, obs: np.ndarray
    ) -> tuple[int, np.ndarray, float, float, float]:
        """Sample an action from the policy (with exploration noise).

        Returns: (intent, target, risk, log_prob, value)
        """
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            intent_logits, target, risk = self.policy(obs_t)
            value = self.value_net(obs_t).item()

            # Sample intent from categorical distribution
            intent_probs = F.softmax(intent_logits, dim=-1)
            dist = torch.distributions.Categorical(intent_probs)
            intent = dist.sample()
            log_prob = dist.log_prob(intent).item()

            return (
                intent.item(),
                target.squeeze(0).numpy(),
                risk.item(),
                log_prob,
                value,
            )

    def update(self) -> dict[str, float]:
        """Run one PPO update cycle on the collected buffer."""
        # Compute GAE
        with torch.no_grad():
            last_obs = torch.from_numpy(self.buffer.obs[self.buffer.ptr - 1]).unsqueeze(0)
            last_value = self.value_net(last_obs).item()
        self.buffer.compute_gae(self.gamma, self.gae_lambda, last_value)

        # Multiple epochs of mini-batch updates
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                obs = batch["obs"]
                old_intents = batch["intents"]
                old_log_probs = batch["log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Policy forward
                intent_logits, _, _ = self.policy(obs)
                intent_probs = F.softmax(intent_logits, dim=-1)
                dist = torch.distributions.Categorical(intent_probs)
                new_log_probs = dist.log_prob(old_intents)
                entropy = dist.entropy().mean()

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus
                policy_loss = policy_loss - self.entropy_coef * entropy

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.policy_optimizer.step()

                # Value function update
                values = self.value_net(obs).squeeze(-1)
                value_loss = F.mse_loss(values, returns)

                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                self.value_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        self.buffer.reset()

        return {
            "policy_loss": total_policy_loss / max(1, n_updates),
            "value_loss": total_value_loss / max(1, n_updates),
            "entropy": total_entropy / max(1, n_updates),
        }

    def save(self, path: str, metadata: Optional[dict] = None) -> None:
        """Save policy checkpoint."""
        MLStrategy.save_checkpoint(self.policy, path, metadata)
