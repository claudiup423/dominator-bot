"""Generate BC dataset from expert policy.

Usage:
    python -m scripts.generate_dataset --episodes 5000 --output data/bc_dataset
"""

from __future__ import annotations

import argparse
import logging

from src.training.behavior_cloning import BCDatasetGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate BC training dataset")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes")
    parser.add_argument("--output", type=str, default="data/bc_dataset", help="Output directory")
    args = parser.parse_args()

    generator = BCDatasetGenerator()
    stats = generator.generate_synthetic(n_episodes=args.episodes, output_path=args.output)
    logger.info(f"Dataset generated: {stats}")


if __name__ == "__main__":
    main()
