"""Deterministic controller: converts tactical plans into precise car controls.

This is the ONLY module that touches throttle/steer/boost/etc.
The ML policy NEVER directly outputs these — it outputs intents and targets,
which get turned into TacticalPlans, which arrive here.

All controllers use simple but robust proportional/PD control.
"""

from __future__ import annotations

import math

from src.types import CarState, ControlOutput, GameState, TacticalPlan, Vec3
from src.utils import (
    angle_to_target,
    distance_2d,
    local_coordinates,
    steer_toward,
    MAX_CAR_SPEED,
)


class Controller:
    """Deterministic low-level controller.

    Executes a TacticalPlan by computing steering, throttle, boost, etc.
    No ML here — pure geometry and PD control.
    """

    # Tuning
    POWERSLIDE_ANGLE = 1.8  # radians — use handbrake above this
    BOOST_SPEED_THRESHOLD = 1400.0  # Don't boost if faster than this (unless plan says so)
    ARRIVAL_SLOWDOWN_DIST = 500.0  # Start slowing when this close

    def execute(self, state: GameState, plan: TacticalPlan) -> ControlOutput:
        """Main entry: produce ControlOutput from a TacticalPlan."""
        ctrl = ControlOutput()

        target = plan.target_position
        car = state.car

        # Distance and angle to target
        dist = distance_2d(car.position, target)
        angle = angle_to_target(car, target)

        # --- Steering ---
        if plan.face_target is not None:
            # We want to face a specific direction (e.g., ball for a shot)
            # But we still drive toward the target position
            ctrl.steer = self._smart_steer(car, target, plan.face_target, dist)
        else:
            ctrl.steer = steer_toward(car, target)

        # --- Handbrake ---
        # Use powerslide for sharp turns when on ground and moving
        if car.is_on_ground and angle > self.POWERSLIDE_ANGLE and car.speed > 400:
            ctrl.handbrake = plan.handbrake or True

        # --- Throttle ---
        ctrl.throttle = self._compute_throttle(car, dist, angle, plan.target_speed)

        # --- Boost ---
        if plan.use_boost and car.boost > 0:
            # Only boost if facing roughly the right direction and not supersonic
            if angle < 0.3 and car.speed < min(plan.target_speed, MAX_CAR_SPEED - 100):
                ctrl.boost = True

        # --- Jump / Dodge ---
        if plan.dodge and car.is_on_ground:
            ctrl.jump = True
            # Dodge direction would need a multi-frame sequence;
            # for now, we just do a simple front-flip trigger
            # (A proper dodge controller would be a state machine)

        if plan.jump:
            ctrl.jump = True

        return ctrl.sanitize()

    def _compute_throttle(
        self, car: CarState, dist: float, angle: float, target_speed: float
    ) -> float:
        """Determine throttle based on distance, angle, and desired speed."""
        # If we need to turn a lot, slow down
        if angle > 1.5:
            return 0.3

        # Slow down near target
        if dist < self.ARRIVAL_SLOWDOWN_DIST and target_speed < MAX_CAR_SPEED:
            speed_ratio = dist / self.ARRIVAL_SLOWDOWN_DIST
            return max(0.2, min(1.0, speed_ratio))

        # Full throttle otherwise
        if car.speed < target_speed:
            return 1.0

        # Coasting / slight brake if overshooting
        if car.speed > target_speed + 200:
            return -0.3

        return 0.6

    def _smart_steer(
        self, car: CarState, drive_target: Vec3, face_target: Vec3, dist: float
    ) -> float:
        """Blend between driving toward target and facing a direction.

        When far: steer toward drive_target.
        When close: steer to face face_target (for shots/clears).
        """
        drive_steer = steer_toward(car, drive_target)
        face_steer = steer_toward(car, face_target)

        # Blend factor: 1.0 = only face_target, 0.0 = only drive_target
        blend = max(0.0, min(1.0, 1.0 - dist / 1500.0))
        return drive_steer * (1 - blend) + face_steer * blend


class RecoveryController:
    """Handles recovery when the car is in an awkward state.

    E.g., upside down, in the air unexpectedly, stuck on wall.
    This is a safety net that the main controller delegates to.
    """

    def execute(self, state: GameState) -> ControlOutput | None:
        """Return recovery controls if needed, or None if no recovery is needed."""
        car = state.car

        # If in the air and not doing an intentional aerial
        if not car.is_on_ground and car.position.z > 100:
            ctrl = ControlOutput()
            # Point wheels down
            ctrl.pitch = -0.5 if car.rotation.pitch > 0.1 else 0.3
            ctrl.roll = -car.rotation.roll * 2.0
            # Boost to land faster if very high
            if car.position.z > 500:
                ctrl.boost = car.boost > 5
            return ctrl.sanitize()

        return None
