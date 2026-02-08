"""League Training: Phase 4 — train against a diverse opponent pool.

Maintains a pool of opponent checkpoints (past selves + scripted styles).
Each episode samples an opponent, preventing exploit-overfitting.
New checkpoints are promoted to the pool based on evaluation gates.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

from src import config


class OpponentPool:
    """Manages a pool of opponent checkpoints for league training."""

    def __init__(self, pool_dir: str = "models/opponent_pool"):
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.pool_dir / "pool_metadata.json"
        self.opponents: list[dict] = []
        self._load_metadata()

    def _load_metadata(self) -> None:
        if self.metadata_path.exists():
            with open(self.metadata_path) as f:
                self.opponents = json.load(f)
        else:
            # Initialize with scripted opponents
            self.opponents = [
                {
                    "name": "expert_conservative",
                    "type": "scripted",
                    "path": None,
                    "aggression": 0.3,
                    "elo": 1000,
                },
                {
                    "name": "expert_aggressive",
                    "type": "scripted",
                    "path": None,
                    "aggression": 0.8,
                    "elo": 1000,
                },
                {
                    "name": "expert_balanced",
                    "type": "scripted",
                    "path": None,
                    "aggression": 0.5,
                    "elo": 1000,
                },
            ]
            self._save_metadata()

    def _save_metadata(self) -> None:
        with open(self.metadata_path, "w") as f:
            json.dump(self.opponents, f, indent=2)

    def add_checkpoint(
        self,
        model_path: str,
        name: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add a trained model checkpoint to the opponent pool."""
        max_pool = config.get("training.league.pool_size", 10)

        # Copy model to pool directory
        dest = self.pool_dir / f"{name}.pt"
        shutil.copy2(model_path, dest)

        entry = {
            "name": name,
            "type": "checkpoint",
            "path": str(dest),
            "elo": 1000,
        }
        if metadata:
            entry.update(metadata)

        self.opponents.append(entry)

        # Prune oldest checkpoints if pool is too large
        checkpoint_opponents = [o for o in self.opponents if o["type"] == "checkpoint"]
        if len(checkpoint_opponents) > max_pool:
            # Remove lowest ELO checkpoint
            worst = min(checkpoint_opponents, key=lambda o: o.get("elo", 0))
            self.opponents.remove(worst)
            worst_path = Path(worst["path"])
            if worst_path.exists():
                worst_path.unlink()

        self._save_metadata()
        logger.info(f"Added opponent '{name}' to pool (size: {len(self.opponents)})")

    def sample_opponent(self) -> dict:
        """Sample an opponent from the pool (weighted by diversity)."""
        if not self.opponents:
            return {"name": "expert_balanced", "type": "scripted", "aggression": 0.5}

        # Uniform random for now — could be weighted by ELO difference
        idx = np.random.randint(len(self.opponents))
        return self.opponents[idx]

    def update_elo(self, name: str, delta: float) -> None:
        """Update an opponent's ELO rating."""
        for opp in self.opponents:
            if opp["name"] == name:
                opp["elo"] = opp.get("elo", 1000) + delta
                break
        self._save_metadata()

    def get_summary(self) -> str:
        """Get a human-readable summary of the pool."""
        lines = [f"Opponent Pool ({len(self.opponents)} opponents):"]
        for opp in self.opponents:
            lines.append(
                f"  {opp['name']}: type={opp['type']}, elo={opp.get('elo', '?')}"
            )
        return "\n".join(lines)


class LeagueTrainer:
    """Orchestrates league training rounds.

    Each round:
    1. Sample opponent from pool
    2. Run N matches
    3. Collect metrics
    4. Update ELO ratings
    5. Optionally add current model to pool
    """

    def __init__(
        self,
        pool_dir: str = "models/opponent_pool",
        output_dir: str = "models/league",
    ):
        self.pool = OpponentPool(pool_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.round_num = 0

    def run_round(self) -> dict:
        """Run one league training round.

        Returns metrics dict for this round.
        """
        self.round_num += 1
        opponent = self.pool.sample_opponent()

        logger.info(
            f"League round {self.round_num}: vs {opponent['name']} "
            f"(type={opponent['type']})"
        )

        # In a full implementation, this would:
        # 1. Set up rlgym-sim env with the sampled opponent
        # 2. Run PPO training for N steps against this opponent
        # 3. Collect win/loss/metrics
        # 4. Update ELO

        # Placeholder metrics
        metrics = {
            "round": self.round_num,
            "opponent": opponent["name"],
            "opponent_type": opponent["type"],
            "matches_played": 0,  # Would be filled by actual training
            "win_rate": 0.0,
            "goals_scored": 0.0,
            "goals_conceded": 0.0,
        }

        # Save round results
        results_path = self.output_dir / f"round_{self.round_num:04d}.json"
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def should_promote(self, metrics: dict) -> bool:
        """Check if current model should be added to opponent pool."""
        threshold = config.get("training.league.promotion_threshold_winrate", 0.55)
        return metrics.get("win_rate", 0) >= threshold
