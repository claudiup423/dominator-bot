"""Train strategy policy via behavior cloning.

Usage:
    python -m scripts.train_bc --dataset data/bc_dataset --epochs 50 --output models/bc
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train strategy policy via BC")
    parser.add_argument("--dataset", type=str, default="data/bc_dataset", help="Dataset path")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--output", type=str, default="models/bc", help="Output directory")
    args = parser.parse_args()

    from src.training.behavior_cloning import BCTrainer

    trainer = BCTrainer(dataset_path=args.dataset)
    trainer.epochs = args.epochs
    results = trainer.train(output_dir=args.output)
    logger.info(f"Training complete: {results}")


if __name__ == "__main__":
    main()
