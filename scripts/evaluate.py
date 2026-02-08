"""Run evaluation matches and check regression gates.

Usage:
    python -m scripts.evaluate --bot expert --opponent expert --matches 20
    python -m scripts.evaluate --bot ml --model models/bc/strategy_model.pt --matches 50
"""

from __future__ import annotations

import argparse
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate bot performance")
    parser.add_argument("--bot", type=str, default="expert", choices=["expert", "ml", "safe"])
    parser.add_argument("--model", type=str, default=None, help="ML model path")
    parser.add_argument("--opponent", type=str, default="expert")
    parser.add_argument("--matches", type=int, default=20)
    parser.add_argument("--output", type=str, default="eval_results")
    parser.add_argument("--baseline", type=str, default=None, help="Baseline metrics JSON for regression check")
    args = parser.parse_args()

    from src.eval import Arena, EvalMetrics

    arena = Arena(output_dir=args.output)
    metrics = arena.evaluate_simulated(
        bot_name=f"DominanceBot({args.bot})",
        opponent_name=args.opponent,
        n_matches=args.matches,
    )

    print(metrics.summary_table())

    # Regression gate check
    if args.baseline:
        from pathlib import Path
        baseline_data = json.loads(Path(args.baseline).read_text())
        baseline = EvalMetrics()
        baseline.win_rate = baseline_data["win_rate"]
        baseline.avg_goals_conceded = baseline_data["avg_goals_conceded"]
        baseline.avg_open_net_conceded = baseline_data["avg_open_net_conceded"]

        passed, failures = arena.check_regression_gates(metrics, baseline)
        if not passed:
            logger.error(f"REGRESSION GATES FAILED: {failures}")
            exit(1)
        else:
            logger.info("All regression gates passed ✓")


if __name__ == "__main__":
    main()
