"""Behavior Cloning (Imitation Learning) training pipeline.

Phase 2 of training: learn to mimic the deterministic expert.

Steps:
1. Run expert in rlgym-sim → collect (obs, intent, target, risk) tuples
2. Train StrategyNetwork with:
   - Cross-entropy loss on intent classification
   - MSE loss on target position regression
   - MSE loss on risk scalar regression
3. Evaluate: compare ML policy's decisions against expert on test set
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
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, random_split

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.types import GameState, Intent, Team
from src.state import StateBuilder
from src.strategy import ExpertStrategy
from src.strategy.ml_policy import StrategyNetwork, MLStrategy, OBS_DIM, NUM_INTENTS
from src import config


class BCDatasetGenerator:
    """Generates imitation learning datasets by running the expert."""

    def __init__(self):
        self.state_builder = StateBuilder(car_index=0, team=Team.BLUE)
        self.expert = ExpertStrategy()

    def generate_from_states(
        self, states: list[dict], output_path: str
    ) -> dict[str, int]:
        """Generate dataset from a list of game state dicts.

        For use without rlgym-sim — takes pre-recorded or synthetic states.
        """
        obs_list = []
        intent_list = []
        target_list = []
        risk_list = []

        for state_data in states:
            game_state = self.state_builder.from_dict(state_data)
            obs = self.state_builder.to_tensor(game_state)
            strategy = self.expert.decide(game_state)

            obs_list.append(obs)
            intent_list.append(strategy.intent.value)
            target_list.append([
                strategy.target.x / 5120.0,  # Normalize to [-1, 1]
                strategy.target.y / 5120.0,
                strategy.target.z / 2000.0,
            ])
            risk_list.append(strategy.risk)

        # Save as numpy arrays
        p = Path(output_path)
        p.mkdir(parents=True, exist_ok=True)
        np.save(p / "obs.npy", np.array(obs_list, dtype=np.float32))
        np.save(p / "intents.npy", np.array(intent_list, dtype=np.int64))
        np.save(p / "targets.npy", np.array(target_list, dtype=np.float32))
        np.save(p / "risks.npy", np.array(risk_list, dtype=np.float32))

        stats = {"total_samples": len(obs_list)}
        with open(p / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Generated BC dataset: {len(obs_list)} samples → {output_path}")
        return stats

    def generate_synthetic(self, n_episodes: int, output_path: str) -> dict[str, int]:
        """Generate synthetic game states for BC training.

        Creates randomized game situations that cover the state space.
        This is useful when rlgym-sim is not available.
        """
        rng = np.random.default_rng(42)
        states = []

        for _ in range(n_episodes):
            # Generate random but plausible game states
            for _ in range(100):  # ~100 ticks per episode
                state = self._random_state(rng)
                states.append(state)

        return self.generate_from_states(states, output_path)

    def _random_state(self, rng: np.random.Generator) -> dict:
        """Create a random but plausible game state dict."""
        return {
            "ball": {
                "position": [
                    rng.uniform(-4000, 4000),
                    rng.uniform(-5000, 5000),
                    rng.uniform(93, 500),
                ],
                "velocity": [
                    rng.uniform(-2000, 2000),
                    rng.uniform(-2000, 2000),
                    rng.uniform(-500, 500),
                ],
            },
            "car": {
                "position": [
                    rng.uniform(-4000, 4000),
                    rng.uniform(-5000, 5000),
                    17.0,
                ],
                "velocity": [
                    rng.uniform(-2000, 2000),
                    rng.uniform(-2000, 2000),
                    0.0,
                ],
                "rotation": [0.0, rng.uniform(-3.14, 3.14), 0.0],
                "boost": rng.uniform(0, 100),
                "is_on_ground": True,
            },
            "opponents": [
                {
                    "position": [
                        rng.uniform(-4000, 4000),
                        rng.uniform(-5000, 5000),
                        17.0,
                    ],
                    "velocity": [
                        rng.uniform(-2000, 2000),
                        rng.uniform(-2000, 2000),
                        0.0,
                    ],
                    "rotation": [0.0, rng.uniform(-3.14, 3.14), 0.0],
                    "boost": rng.uniform(0, 100),
                    "is_on_ground": True,
                }
            ],
        }


class BCTrainer:
    """Trains a StrategyNetwork via behavior cloning."""

    def __init__(self, dataset_path: str):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for training")

        self.dataset_path = Path(dataset_path)
        self.model = StrategyNetwork()

        # Load config
        self.lr = config.get("training.bc.learning_rate", 0.001)
        self.batch_size = config.get("training.bc.batch_size", 256)
        self.epochs = config.get("training.bc.epochs", 50)
        self.val_split = config.get("training.bc.validation_split", 0.1)

    def load_dataset(self) -> tuple:
        """Load dataset from numpy files."""
        obs = np.load(self.dataset_path / "obs.npy")
        intents = np.load(self.dataset_path / "intents.npy")
        targets = np.load(self.dataset_path / "targets.npy")
        risks = np.load(self.dataset_path / "risks.npy")

        obs_t = torch.from_numpy(obs)
        intents_t = torch.from_numpy(intents)
        targets_t = torch.from_numpy(targets)
        risks_t = torch.from_numpy(risks).unsqueeze(-1)

        dataset = TensorDataset(obs_t, intents_t, targets_t, risks_t)

        # Split
        val_size = int(len(dataset) * self.val_split)
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        logger.info(f"Loaded dataset: {train_size} train, {val_size} val samples")
        return train_loader, val_loader

    def train(self, output_dir: str = "models/bc") -> dict:
        """Run BC training loop."""
        train_loader, val_loader = self.load_dataset()

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        intent_loss_fn = nn.CrossEntropyLoss()
        target_loss_fn = nn.MSELoss()
        risk_loss_fn = nn.MSELoss()

        best_val_loss = float("inf")
        history = []

        for epoch in range(self.epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            train_intent_acc = 0.0
            n_batches = 0

            for obs, intents, targets, risks in train_loader:
                optimizer.zero_grad()
                intent_logits, pred_targets, pred_risks = self.model(obs)

                loss_intent = intent_loss_fn(intent_logits, intents)
                loss_target = target_loss_fn(pred_targets, targets)
                loss_risk = risk_loss_fn(pred_risks, risks)
                loss = loss_intent + loss_target * 5.0 + loss_risk

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pred_intents = intent_logits.argmax(dim=-1)
                train_intent_acc += (pred_intents == intents).float().mean().item()
                n_batches += 1

            train_loss /= max(1, n_batches)
            train_intent_acc /= max(1, n_batches)

            # Validate
            self.model.eval()
            val_loss = 0.0
            val_intent_acc = 0.0
            n_val = 0
            with torch.no_grad():
                for obs, intents, targets, risks in val_loader:
                    intent_logits, pred_targets, pred_risks = self.model(obs)
                    loss_intent = intent_loss_fn(intent_logits, intents)
                    loss_target = target_loss_fn(pred_targets, targets)
                    loss_risk = risk_loss_fn(pred_risks, risks)
                    loss = loss_intent + loss_target * 5.0 + loss_risk
                    val_loss += loss.item()
                    pred_intents = intent_logits.argmax(dim=-1)
                    val_intent_acc += (pred_intents == intents).float().mean().item()
                    n_val += 1

            val_loss /= max(1, n_val)
            val_intent_acc /= max(1, n_val)

            epoch_data = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "train_intent_acc": round(train_intent_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_intent_acc": round(val_intent_acc, 4),
            }
            history.append(epoch_data)

            logger.info(
                f"Epoch {epoch+1}/{self.epochs}: "
                f"train_loss={train_loss:.4f} intent_acc={train_intent_acc:.4f} | "
                f"val_loss={val_loss:.4f} intent_acc={val_intent_acc:.4f}"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                out_dir = Path(output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                MLStrategy.save_checkpoint(
                    self.model,
                    str(out_dir / "strategy_model.pt"),
                    metadata={"epoch": epoch + 1, "val_loss": val_loss},
                )

        # Save training history
        out_dir = Path(output_dir)
        with open(out_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        return {
            "best_val_loss": best_val_loss,
            "final_train_acc": history[-1]["train_intent_acc"] if history else 0,
            "final_val_acc": history[-1]["val_intent_acc"] if history else 0,
        }
