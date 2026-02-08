"""Kickoff handler: speed flip kickoff for fast ball contact.

A speed flip is a diagonal dodge-cancel that gets to the ball faster than
any other kickoff. The sequence is:
1. Boost + throttle forward
2. Jump
3. Diagonal dodge (pitch down + yaw to side)
4. Cancel the flip (pitch up) to stay flat
5. Continue boosting into the ball

Timed in ticks at 120Hz.
"""

from __future__ import annotations

import math

from src.types import GameState, StrategyOutput, ControlOutput, Intent, Vec3


# Speed flip timing constants (at 120Hz)
PHASE_BOOST_START = 0
PHASE_FIRST_JUMP = 12
PHASE_DODGE = 24
PHASE_CANCEL_START = 28
PHASE_CANCEL_END = 48
PHASE_DRIVE_TO_BALL = 60


class KickoffHandler:
    """Handles kickoff with a speed flip for fast ball contact."""

    def __init__(self):
        self._kickoff_active = False
        self._kickoff_tick = 0
        self._dodge_direction = 1
        self._has_jumped = False
        self._has_dodged = False

    def is_kickoff(self, state: GameState) -> bool:
        return state.is_kickoff

    def decide(self, state: GameState) -> StrategyOutput:
        if not self._kickoff_active:
            self._kickoff_active = True
            self._kickoff_tick = 0
            self._has_jumped = False
            self._has_dodged = False
            self._dodge_direction = 1 if state.car.position.x < 0 else -1

        self._kickoff_tick += 1

        return StrategyOutput(
            intent=Intent.CHALLENGE,
            target=Vec3(0, 0, 0),
            risk=1.0,
        )

    def get_controls(self, state: GameState) -> ControlOutput | None:
        """Get direct controls for the speed flip kickoff."""
        if not self._kickoff_active:
            return None

        tick = self._kickoff_tick
        ctrl = ControlOutput()

        ctrl.throttle = 1.0
        ctrl.boost = True

        if tick < PHASE_FIRST_JUMP:
            from src.utils import steer_toward
            ctrl.steer = steer_toward(state.car, Vec3(0, 0, 0))

        elif tick == PHASE_FIRST_JUMP:
            ctrl.jump = True
            self._has_jumped = True

        elif PHASE_FIRST_JUMP < tick < PHASE_DODGE:
            ctrl.jump = False
            ctrl.pitch = -0.3

        elif tick == PHASE_DODGE:
            ctrl.jump = True
            ctrl.pitch = -1.0
            ctrl.yaw = 0.9 * self._dodge_direction
            self._has_dodged = True

        elif PHASE_CANCEL_START <= tick < PHASE_CANCEL_END:
            ctrl.jump = False
            ctrl.pitch = 1.0
            ctrl.yaw = 0.0
            ctrl.roll = -0.3 * self._dodge_direction

        else:
            ctrl.jump = False
            ctrl.pitch = 0.0
            from src.utils import steer_toward
            ctrl.steer = steer_toward(state.car, Vec3(0, 0, 0))

        return ctrl.sanitize()

    def reset(self) -> None:
        self._kickoff_active = False
        self._kickoff_tick = 0
        self._has_jumped = False
        self._has_dodged = False
