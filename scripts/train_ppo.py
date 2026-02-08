"""Train strategy policy via PPO reinforcement learning.

Usage:
    python -m scripts.train_ppo --timesteps 50000000 --resume models/bc/strategy_model.pt
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train strategy policy via PPO")
    parser.add_argument("--timesteps", type=int, default=50_000_000, help="Total timesteps")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--output", type=str, default="models/ppo", help="Output directory")
    args = parser.parse_args()

    from pathlib import Path
    from src.training.ppo import PPOTrainer

    trainer = PPOTrainer(model_path=args.resume)

    # Training loop outline (requires rlgym-sim environment)
    logger.info(f"PPO training: {args.timesteps} timesteps")
    logger.info("NOTE: Full PPO training requires rlgym-sim environment.")
    logger.info("This script provides the training loop skeleton.")
    logger.info("To train, install rlgym-sim and integrate with the PPOTrainer.")

    # Save initial model
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save(str(out / "strategy_model.pt"), metadata={"timesteps": 0})
    logger.info(f"Saved initial model to {out / 'strategy_model.pt'}")


if __name__ == "__main__":
    main()
