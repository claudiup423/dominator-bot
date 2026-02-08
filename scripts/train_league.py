"""Run league training against opponent pool.

Usage:
    python -m scripts.train_league --rounds 100
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="League training")
    parser.add_argument("--rounds", type=int, default=100, help="Number of rounds")
    parser.add_argument("--pool-dir", type=str, default="models/opponent_pool")
    parser.add_argument("--output", type=str, default="models/league")
    args = parser.parse_args()

    from src.training.league import LeagueTrainer

    trainer = LeagueTrainer(pool_dir=args.pool_dir, output_dir=args.output)

    logger.info(f"Starting league training: {args.rounds} rounds")
    logger.info(trainer.pool.get_summary())

    for r in range(args.rounds):
        metrics = trainer.run_round()
        logger.info(f"Round {r+1}: {metrics}")

        if trainer.should_promote(metrics):
            logger.info(f"Promoting model from round {r+1}")

    logger.info("League training complete")


if __name__ == "__main__":
    main()
