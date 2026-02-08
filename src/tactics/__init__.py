"""Tactics Planner: converts high-level strategy (intent + target + risk) into
a concrete TacticalPlan that the Controller can execute.

This module bridges the gap between "what to do" (strategy) and "how to do it"
(controller). It computes specific positions, speeds, and modes based on the
current intent and game state.
"""

from __future__ import annotations

import math

from src.types import (
    GameState,
    Intent,
    StrategyOutput,
    TacticalPlan,
    Vec3,
)
from src.utils import (
    back_post_position,
    clamp_to_field,
    distance_2d,
    estimate_intercept_time,
    predict_ball_position,
    FIELD_HALF_LENGTH,
    FIELD_HALF_WIDTH,
    GOAL_HALF_WIDTH,
    MAX_CAR_SPEED,
)


class TacticsPlanner:
    """Translates strategy outputs into executable tactical plans."""

    def plan(self, state: GameState, strategy: StrategyOutput) -> TacticalPlan:
        """Main entry: route to the appropriate tactic handler."""
        intent = strategy.intent
        risk = strategy.risk

        match intent:
            case Intent.DEFEND_SHADOW:
                return self._plan_shadow(state, strategy)
            case Intent.DEFEND_CLEAR:
                return self._plan_clear(state, strategy)
            case Intent.CHALLENGE:
                return self._plan_challenge(state, strategy)
            case Intent.ATTACK_SHOOT:
                return self._plan_shoot(state, strategy)
            case Intent.ATTACK_POSSESSION:
                return self._plan_possession(state, strategy)
            case Intent.ROTATE_BACK:
                return self._plan_rotate(state, strategy)
            case Intent.GRAB_BOOST_SAFE:
                return self._plan_grab_boost(state, strategy)
            case _:
                # Fallback: rotate back
                return self._plan_rotate(state, strategy)

    def _plan_shadow(self, state: GameState, strategy: StrategyOutput) -> TacticalPlan:
        """Shadow defense: stay between ball and our goal."""
        shadow_offset = 800.0 * (1.0 - strategy.risk * 0.5)  # Tighter shadow when aggressive

        ball_to_goal = (state.own_goal - state.ball.position).flat().normalized()
        shadow_pos = state.ball.position + ball_to_goal * shadow_offset

        # Clamp to reasonable area (don't go into our own net)
        shadow_pos = clamp_to_field(shadow_pos, margin=300.0)

        # Match ball speed for proper shadowing
        target_speed = min(MAX_CAR_SPEED, state.ball.velocity.length() + 300)

        return TacticalPlan(
            target_position=shadow_pos,
            target_speed=target_speed,
            face_target=state.ball.position,
            use_boost=state.car.boost > 30 and target_speed > 1500,
            source_intent=Intent.DEFEND_SHADOW,
        )

    def _plan_clear(self, state: GameState, strategy: StrategyOutput) -> TacticalPlan:
        """Clear the ball from danger: intercept and hit away from our goal."""
        intercept_time, intercept_pos = estimate_intercept_time(
            state.car, state.ball, max_time=3.0
        )

        # Aim the clear toward the opponent's half
        clear_direction = (state.opp_goal - intercept_pos).flat().normalized()
        face_target = intercept_pos + clear_direction * 1000

        return TacticalPlan(
            target_position=intercept_pos,
            target_speed=MAX_CAR_SPEED,
            face_target=face_target,
            use_boost=True,
            source_intent=Intent.DEFEND_CLEAR,
        )

    def _plan_challenge(self, state: GameState, strategy: StrategyOutput) -> TacticalPlan:
        """Challenge: go directly to the ball for a 50/50."""
        intercept_time, intercept_pos = estimate_intercept_time(
            state.car, state.ball, max_time=2.0
        )

        return TacticalPlan(
            target_position=intercept_pos,
            target_speed=MAX_CAR_SPEED,
            face_target=state.ball.position,
            use_boost=True,
            dodge=intercept_time < 0.5,  # Front-flip into the ball if close
            source_intent=Intent.CHALLENGE,
        )

    def _plan_shoot(self, state: GameState, strategy: StrategyOutput) -> TacticalPlan:
        """Shoot on goal: intercept ball while aiming at opponent's goal."""
        intercept_time, intercept_pos = estimate_intercept_time(
            state.car, state.ball, max_time=3.0
        )

        # Aim at the far post for best shot
        ball_x = state.ball.position.x
        if ball_x > 0:
            aim_x = -GOAL_HALF_WIDTH * 0.6  # Aim left post
        else:
            aim_x = GOAL_HALF_WIDTH * 0.6  # Aim right post

        aim_target = Vec3(aim_x, state.opp_goal.y, 0.0)

        return TacticalPlan(
            target_position=intercept_pos,
            target_speed=MAX_CAR_SPEED,
            face_target=aim_target,
            use_boost=True,
            source_intent=Intent.ATTACK_SHOOT,
        )

    def _plan_possession(self, state: GameState, strategy: StrategyOutput) -> TacticalPlan:
        """Possession play: get to the ball and keep it controlled."""
        # Get to the ball from behind (relative to opponent goal) for better control
        ball_to_opp = (state.opp_goal - state.ball.position).flat().normalized()
        approach_pos = state.ball.position - ball_to_opp * 200  # Behind the ball

        approach_pos = clamp_to_field(approach_pos)

        return TacticalPlan(
            target_position=approach_pos,
            target_speed=min(1800.0, state.ball.velocity.length() + 500),
            face_target=state.ball.position,
            use_boost=state.car.boost > 40,
            source_intent=Intent.ATTACK_POSSESSION,
        )

    def _plan_rotate(self, state: GameState, strategy: StrategyOutput) -> TacticalPlan:
        """Rotate back to defensive position (back post)."""
        bp = back_post_position(
            state.own_goal, state.ball.position, offset=400.0
        )

        return TacticalPlan(
            target_position=bp,
            target_speed=MAX_CAR_SPEED,
            face_target=state.ball.position,
            use_boost=state.car.boost > 20,
            source_intent=Intent.ROTATE_BACK,
        )

    def _plan_grab_boost(self, state: GameState, strategy: StrategyOutput) -> TacticalPlan:
        """Grab the nearest safe boost pad."""
        # Find the best boost pad: big > small, closer > farther, safe path
        best_pad = None
        best_score = -1.0

        for pad in state.boost_pads:
            if not pad.is_active:
                continue

            dist = distance_2d(state.car.position, pad.position)
            # Score: prefer big pads, close pads, and pads on our rotation path
            size_bonus = 2.0 if pad.is_big else 1.0
            dist_score = max(0.0, 1.0 - dist / 5000.0)

            # Safety: prefer pads that are on the way back to our goal
            pad_to_goal = distance_2d(pad.position, state.own_goal)
            car_to_goal = distance_2d(state.car.position, state.own_goal)
            # Pad is "on the way" if it doesn't take us farther from goal
            path_safety = 1.0 if pad_to_goal <= car_to_goal + 500 else 0.3

            score = size_bonus * dist_score * path_safety
            if score > best_score:
                best_score = score
                best_pad = pad

        if best_pad is not None:
            target = best_pad.position
        else:
            # No good pad found, just rotate back
            target = back_post_position(state.own_goal, state.ball.position)

        return TacticalPlan(
            target_position=target,
            target_speed=MAX_CAR_SPEED,
            face_target=None,
            use_boost=False,  # Don't burn boost to get boost
            source_intent=Intent.GRAB_BOOST_SAFE,
        )
