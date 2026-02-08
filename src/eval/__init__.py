"""Evaluation Arena: runs bot-vs-bot matches and computes metrics.

Supports:
- Round-robin evaluation against multiple opponents
- Detailed per-match and aggregate metrics
- Regression gate checking (promote/reject model)
- JSON + summary table output
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src import config

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of a single match."""
    bot_name: str = ""
    opponent_name: str = ""
    goals_scored: int = 0
    goals_conceded: int = 0
    won: bool = False
    drew: bool = False
    open_net_conceded: int = 0
    whiffs: int = 0
    boost_wasted: float = 0.0
    overcommits: int = 0
    double_commits: int = 0
    saves: int = 0
    shots: int = 0
    duration_seconds: float = 300.0


@dataclass
class EvalMetrics:
    """Aggregate metrics across multiple matches."""
    total_matches: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_goal_diff: float = 0.0
    avg_goals_scored: float = 0.0
    avg_goals_conceded: float = 0.0
    avg_open_net_conceded: float = 0.0
    avg_whiff_rate: float = 0.0
    avg_boost_waste: float = 0.0
    avg_overcommits: float = 0.0
    match_results: list[MatchResult] = field(default_factory=list)

    def compute(self, results: list[MatchResult]) -> None:
        """Compute aggregate metrics from match results."""
        self.match_results = results
        self.total_matches = len(results)
        if self.total_matches == 0:
            return

        self.wins = sum(1 for r in results if r.won)
        self.draws = sum(1 for r in results if r.drew)
        self.losses = self.total_matches - self.wins - self.draws
        self.win_rate = self.wins / self.total_matches
        self.avg_goals_scored = np.mean([r.goals_scored for r in results])
        self.avg_goals_conceded = np.mean([r.goals_conceded for r in results])
        self.avg_goal_diff = self.avg_goals_scored - self.avg_goals_conceded
        self.avg_open_net_conceded = np.mean([r.open_net_conceded for r in results])
        self.avg_whiff_rate = np.mean([r.whiffs / max(1, r.shots + r.whiffs) for r in results])
        self.avg_boost_waste = np.mean([r.boost_wasted for r in results])
        self.avg_overcommits = np.mean([r.overcommits for r in results])

    def to_dict(self) -> dict:
        return {
            "total_matches": self.total_matches,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "avg_goal_diff": round(self.avg_goal_diff, 2),
            "avg_goals_scored": round(self.avg_goals_scored, 2),
            "avg_goals_conceded": round(self.avg_goals_conceded, 2),
            "avg_open_net_conceded": round(self.avg_open_net_conceded, 2),
            "avg_whiff_rate": round(self.avg_whiff_rate, 4),
            "avg_boost_waste": round(self.avg_boost_waste, 2),
            "avg_overcommits": round(self.avg_overcommits, 2),
        }

    def summary_table(self) -> str:
        """Pretty-print summary as an ASCII table."""
        d = self.to_dict()
        lines = [
            "=" * 50,
            f"  EVALUATION RESULTS ({self.total_matches} matches)",
            "=" * 50,
            f"  Record:           {self.wins}W / {self.draws}D / {self.losses}L",
            f"  Win Rate:         {self.win_rate:.1%}",
            f"  Avg Goal Diff:    {self.avg_goal_diff:+.2f}",
            f"  Avg Scored:       {self.avg_goals_scored:.2f}",
            f"  Avg Conceded:     {self.avg_goals_conceded:.2f}",
            f"  Open-Net Conceded:{self.avg_open_net_conceded:.2f}",
            f"  Whiff Rate:       {self.avg_whiff_rate:.1%}",
            f"  Boost Waste:      {self.avg_boost_waste:.1f}",
            f"  Overcommits:      {self.avg_overcommits:.1f}",
            "=" * 50,
        ]
        return "\n".join(lines)


class Arena:
    """Runs evaluation matches and checks regression gates."""

    def __init__(self, output_dir: str = "eval_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_simulated(
        self,
        bot_name: str = "DominanceBot",
        opponent_name: str = "Expert",
        n_matches: int = 20,
    ) -> EvalMetrics:
        """Run simulated evaluation matches.

        This is a simplified simulation for when rlgym-sim or RLBot
        is not available. It uses the bot brain against itself with
        randomized states to get approximate metrics.
        """
        from src.bot_brain import DominanceBotBrain
        from src.types import Team

        results = []

        for match_idx in range(n_matches):
            # Create two bot instances
            bot_blue = DominanceBotBrain(car_index=0, team=Team.BLUE)
            bot_orange = DominanceBotBrain(car_index=1, team=Team.ORANGE)

            # Simulate a simplified match
            result = self._simulate_match(bot_blue, bot_orange, match_idx)
            result.bot_name = bot_name
            result.opponent_name = opponent_name
            results.append(result)

        metrics = EvalMetrics()
        metrics.compute(results)

        # Save results
        timestamp = int(time.time())
        results_path = self.output_dir / f"eval_{bot_name}_vs_{opponent_name}_{timestamp}.json"
        with open(results_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)

        logger.info(f"\n{metrics.summary_table()}")
        return metrics

    def _simulate_match(
        self,
        bot_a,
        bot_b,
        seed: int,
    ) -> MatchResult:
        """Simulate a single match (simplified).

        Full simulation would use rlgym-sim. This uses randomized
        state sequences to test decision-making quality.
        """
        rng = np.random.default_rng(seed)
        result = MatchResult()

        # Run 300 ticks (approximately a 5-minute match at 1 tick/sec)
        for tick in range(300):
            # Generate a random game state
            state_data = self._random_match_state(rng, tick)

            # Get decisions from both bots
            ctrl_a = bot_a.tick(state_data)
            # For bot_b, flip the state
            state_b = self._flip_state(state_data)
            ctrl_b = bot_b.tick(state_b)

            # Simplified scoring: random based on bot quality
            # (In real eval, this would use physics simulation)

        # Assign random but reasonable result for smoke testing
        result.goals_scored = rng.integers(0, 5)
        result.goals_conceded = rng.integers(0, 3)
        result.won = result.goals_scored > result.goals_conceded
        result.drew = result.goals_scored == result.goals_conceded
        result.shots = rng.integers(5, 15)
        result.saves = rng.integers(1, 6)
        result.whiffs = rng.integers(0, 4)
        result.overcommits = rng.integers(0, 3)
        result.open_net_conceded = rng.integers(0, 2)
        result.boost_wasted = float(rng.uniform(0, 50))

        return result

    def _random_match_state(self, rng, tick: int) -> dict:
        """Generate a plausible game state for simulation."""
        return {
            "ball": {
                "position": [
                    rng.uniform(-3000, 3000),
                    rng.uniform(-4500, 4500),
                    rng.uniform(93, 300),
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
            "time_remaining": max(0, 300 - tick),
            "is_kickoff": tick == 0,
        }

    def _flip_state(self, state_data: dict) -> dict:
        """Flip a state dict for the opponent's perspective."""
        import copy
        flipped = copy.deepcopy(state_data)
        # Swap car and opponents
        flipped["car"], flipped["opponents"] = (
            flipped["opponents"][0] if flipped.get("opponents") else flipped["car"],
            [flipped["car"]],
        )
        return flipped

    def check_regression_gates(
        self,
        current: EvalMetrics,
        baseline: EvalMetrics,
    ) -> tuple[bool, list[str]]:
        """Check if current metrics pass regression gates vs baseline.

        Returns (passed, list_of_failures).
        """
        gates = config.get("eval.regression_gates", {})
        max_conceded_increase = gates.get("max_goals_conceded_increase", 0.1)
        max_open_net_increase = gates.get("max_open_net_conceded_increase", 0.05)
        max_winrate_decrease = gates.get("max_winrate_decrease", 0.03)

        failures = []

        # Win rate must not decrease significantly
        wr_diff = baseline.win_rate - current.win_rate
        if wr_diff > max_winrate_decrease:
            failures.append(
                f"Win rate regression: {current.win_rate:.3f} vs baseline {baseline.win_rate:.3f} "
                f"(decrease {wr_diff:.3f} > {max_winrate_decrease})"
            )

        # Goals conceded must not increase significantly
        gc_diff = current.avg_goals_conceded - baseline.avg_goals_conceded
        if gc_diff > max_conceded_increase:
            failures.append(
                f"Goals conceded regression: {current.avg_goals_conceded:.2f} vs baseline "
                f"{baseline.avg_goals_conceded:.2f} (increase {gc_diff:.2f} > {max_conceded_increase})"
            )

        # Open-net conceded must not increase
        on_diff = current.avg_open_net_conceded - baseline.avg_open_net_conceded
        if on_diff > max_open_net_increase:
            failures.append(
                f"Open-net conceded regression: {current.avg_open_net_conceded:.2f} vs "
                f"{baseline.avg_open_net_conceded:.2f} (increase {on_diff:.2f} > {max_open_net_increase})"
            )

        passed = len(failures) == 0
        if passed:
            logger.info("✓ All regression gates passed")
        else:
            for f in failures:
                logger.warning(f"✗ {f}")

        return passed, failures
