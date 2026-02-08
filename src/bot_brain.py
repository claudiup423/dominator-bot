"""Bot Brain: the main orchestrator that runs the full tick pipeline.

Pipeline per tick:
1. StateBuilder.from_dict(packet) → GameState
2. KickoffHandler.decide() or StrategyPolicy.decide() → StrategyOutput
3. TacticsPlanner.plan() → TacticalPlan
4. SafetySupervisor.check() → SafeTacticalPlan
5. Controller.execute() → ControlOutput

This module ties everything together and handles mode switching
(expert vs ML vs safe fallback).
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, Optional

from src.types import ControlOutput, GameState, Team
from src.state import StateBuilder
from src.strategy import ExpertStrategy
from src.strategy.ml_policy import MLStrategy
from src.tactics import TacticsPlanner
from src.safety import SafetySupervisor
from src.control import Controller, RecoveryController
from src.kickoff import KickoffHandler
from src import config

logger = logging.getLogger(__name__)


class BotMode(Enum):
    EXPERT = "expert"
    ML = "ml"
    SAFE = "safe"  # Fallback: pure deterministic, extra conservative


class DominanceBotBrain:
    """Main bot brain — call `tick()` each frame with the game packet."""

    def __init__(
        self,
        car_index: int = 0,
        team: Team = Team.BLUE,
        mode: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        # Determine mode
        mode_str = mode or config.get("bot.mode", "expert")
        self.mode = BotMode(mode_str)

        # Pipeline components
        self.state_builder = StateBuilder(car_index=car_index, team=team)
        self.expert = ExpertStrategy()
        self.ml_strategy = MLStrategy(model_path=model_path) if model_path else MLStrategy()
        self.tactics = TacticsPlanner()
        self.safety = SafetySupervisor()
        self.controller = Controller()
        self.recovery = RecoveryController()
        self.kickoff = KickoffHandler()

        # Performance tracking
        self._tick_times: list[float] = []
        self._total_ticks = 0
        self._override_count = 0

        # If ML mode but model isn't loaded, fall back to expert
        if self.mode == BotMode.ML and not self.ml_strategy.is_ready:
            logger.warning("ML model not loaded — falling back to expert mode")
            self.mode = BotMode.EXPERT

        logger.info(f"DominanceBotBrain initialized in {self.mode.value} mode")

    def tick(self, packet_data: dict[str, Any]) -> ControlOutput:
        """Process one game tick. This is the main entry point.

        Args:
            packet_data: Dict-format game data (from RLBot adapter or rlgym)

        Returns:
            ControlOutput with all controller values
        """
        start = time.perf_counter()

        try:
            # 1. Build state
            state = self.state_builder.from_dict(packet_data)

            # 2. Check for recovery (car in bad state)
            recovery_ctrl = self.recovery.execute(state)
            if recovery_ctrl is not None:
                return recovery_ctrl

            # 3. Strategy decision
            if self.kickoff.is_kickoff(state):
                strategy = self.kickoff.decide(state)
                # Speed flip kickoff has its own controls — bypass the normal pipeline
                kickoff_ctrl = self.kickoff.get_controls(state)
                if kickoff_ctrl is not None:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    self._tick_times.append(elapsed_ms)
                    self._total_ticks += 1
                    return kickoff_ctrl
            else:
                self.kickoff.reset()
                strategy = self._get_strategy(state)

            # 4. Tactical planning
            plan = self.tactics.plan(state, strategy)

            # 5. Safety check
            safe_plan = self.safety.check(state, strategy, plan)
            if safe_plan.was_overridden:
                self._override_count += 1

            # 6. Execute controls
            ctrl = self.controller.execute(state, safe_plan)

            # Track performance
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._tick_times.append(elapsed_ms)
            self._total_ticks += 1

            if self._total_ticks % 1000 == 0:
                avg = sum(self._tick_times[-1000:]) / min(1000, len(self._tick_times))
                logger.debug(
                    f"Tick {self._total_ticks}: avg={avg:.2f}ms, "
                    f"overrides={self._override_count}"
                )

            return ctrl

        except Exception as e:
            logger.error(f"Tick failed: {e}", exc_info=True)
            # Ultimate fallback: do nothing harmful
            return ControlOutput(throttle=0.3, steer=0.0)

    def _get_strategy(self, state: GameState):
        """Get strategy output based on current mode."""
        if self.mode == BotMode.ML and self.ml_strategy.is_ready:
            obs = self.state_builder.to_tensor(state)
            result = self.ml_strategy.decide(obs)
            if result is not None:
                return result
            # ML failed → fall back to expert for this tick
            logger.warning("ML inference failed, using expert fallback")

        # Expert or safe mode
        return self.expert.decide(state)

    def get_stats(self) -> dict:
        """Return performance statistics."""
        recent = self._tick_times[-1000:] if self._tick_times else [0]
        return {
            "total_ticks": self._total_ticks,
            "avg_tick_ms": sum(recent) / max(1, len(recent)),
            "max_tick_ms": max(recent) if recent else 0,
            "override_count": self._override_count,
            "override_rate": self._override_count / max(1, self._total_ticks),
            "mode": self.mode.value,
        }
