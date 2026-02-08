"""Run the bot locally for testing (without RLBot).

Creates a DominanceBot instance and runs it through some simulated ticks
to verify the pipeline works end-to-end.

Usage:
    python -m scripts.run_bot
    python -m scripts.run_bot --mode ml --model models/bc/strategy_model.pt
"""

from __future__ import annotations

import argparse
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run bot locally for testing")
    parser.add_argument("--mode", type=str, default="expert", choices=["expert", "ml", "safe"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--ticks", type=int, default=1000)
    args = parser.parse_args()

    from src.bot_brain import DominanceBotBrain
    from src.types import Team

    import numpy as np

    brain = DominanceBotBrain(
        car_index=0,
        team=Team.BLUE,
        mode=args.mode,
        model_path=args.model,
    )

    rng = np.random.default_rng(42)
    logger.info(f"Running {args.ticks} ticks in {args.mode} mode...")

    start = time.perf_counter()
    for tick in range(args.ticks):
        # Generate a plausible game state
        state = {
            "ball": {
                "position": [
                    rng.uniform(-3000, 3000),
                    rng.uniform(-4500, 4500),
                    93.0 + rng.uniform(0, 200),
                ],
                "velocity": [
                    rng.uniform(-1500, 1500),
                    rng.uniform(-1500, 1500),
                    rng.uniform(-200, 200),
                ],
            },
            "car": {
                "position": [rng.uniform(-3000, 3000), rng.uniform(-4500, 4500), 17.0],
                "velocity": [rng.uniform(-1000, 1000), rng.uniform(-1000, 1000), 0.0],
                "rotation": [0.0, rng.uniform(-3.14, 3.14), 0.0],
                "boost": rng.uniform(0, 100),
                "is_on_ground": True,
            },
            "opponents": [{
                "position": [rng.uniform(-3000, 3000), rng.uniform(-4500, 4500), 17.0],
                "velocity": [rng.uniform(-1000, 1000), rng.uniform(-1000, 1000), 0.0],
                "rotation": [0.0, rng.uniform(-3.14, 3.14), 0.0],
                "boost": rng.uniform(0, 100),
                "is_on_ground": True,
            }],
            "time_remaining": max(0, 300 - tick * 0.3),
        }

        ctrl = brain.tick(state)

    elapsed = time.perf_counter() - start
    stats = brain.get_stats()

    logger.info(f"Completed {args.ticks} ticks in {elapsed:.3f}s")
    logger.info(f"Average tick: {stats['avg_tick_ms']:.3f}ms")
    logger.info(f"Max tick: {stats['max_tick_ms']:.3f}ms")
    logger.info(f"Override rate: {stats['override_rate']:.1%}")
    logger.info(f"Mode: {stats['mode']}")

    # Verify latency budget
    budget = 2.0  # ms
    if stats['avg_tick_ms'] < budget:
        logger.info(f"✓ Latency within budget ({stats['avg_tick_ms']:.3f}ms < {budget}ms)")
    else:
        logger.warning(f"✗ Latency EXCEEDS budget ({stats['avg_tick_ms']:.3f}ms > {budget}ms)")


if __name__ == "__main__":
    main()
